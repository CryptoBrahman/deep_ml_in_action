# pip install torch torchtext
# python -m spacy download en

import torch
import torchtext
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

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) if self.lstm.bidirectional else hidden[-1, :, :])
        return self.fc(hidden)

# Instantiate the model
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1  # Change for multiclass classification
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Change for multiclass classification
optimizer = torch.optim.Adam(model.parameters())

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    for batch in iterator:
        text, text_lengths = batch.text
        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
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
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
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
