import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator

data = {
    'text': ['This is a positive example.', 'Negative sentiment here.', 'Another positive statement.'],
    'label': [1, 0, 1]
}

df = pd.DataFrame(data)

# Define fields for text and label
TEXT = Field(tokenize='spacy', lower=True, include_lengths=True)
LABEL = LabelField(dtype=torch.float)

# Create a torchtext dataset
fields = [('text', TEXT), ('label', LABEL)]
train_data, test_data = TabularDataset.splits(path='.', train='train.csv', test='test.csv', format='csv', fields=fields)

# Build vocabulary
TEXT.build_vocab(train_data, max_size=10000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

# Create iterators for training and testing sets
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=8,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# CNN model
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Instantiate the model
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1  # Change for multiclass classification
DROPOUT = 0.5

model = CNNClassifier(VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Change for multiclass classification
optimizer = torch.optim.Adam(model.parameters())

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    for batch in iterator:
        text, _ = batch.text
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

# Evaluation loop
def evaluate(model, iterator, criterion):
    model.eval()
    correct_predictions = 0
    total_loss = 0
    with torch.no_grad():
        for batch in iterator:
            text, _ = batch.text
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, batch.label)
            total_loss += loss.item()
            rounded_predictions = torch.round(torch.sigmoid(predictions))
            correct_predictions += (rounded_predictions == batch.label).sum().item()

    return correct_predictions / len(iterator.dataset), total_loss / len(iterator)

# Train the model
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train(model, train_iterator, optimizer, criterion)
    train_acc, train_loss = evaluate(model, train_iterator, criterion)
    test_acc, test_loss = evaluate(model, test_iterator, criterion)
    print(f'Epoch: {epoch+1}, Train Acc: {train_acc:.3f}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.3f}, Test Loss: {test_loss:.4f}')
