# pip install torch torchtext

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import GloVe


# Define the neural network model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        output = self.sigmoid(output)
        return output


# Define hyperparameters
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 1

# Instantiate the model
model = SentimentModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# Download and load the IMDB dataset
TEXT = data.Field(tokenize='spacy', include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# Build the vocabulary
TEXT.build_vocab(train_data, max_size=vocab_size, vectors=GloVe(name='6B', dim=embedding_dim))
LABEL.build_vocab(train_data)

# Create iterators for the training and testing sets
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=64,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 5
for epoch in range(num_epochs):
    for batch in train_iterator:
        text, text_lengths = batch.text
        labels = batch.label.unsqueeze(1)

        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

# Testing the model
model.eval()
with torch.no_grad():
    for batch in test_iterator:
        text, text_lengths = batch.text
        labels = batch.label.unsqueeze(1)

        predictions = model(text)
        predictions = (predictions > 0.5).float()

        accuracy = torch.sum(predictions == labels).item() / len(labels)
        print(f'Test Accuracy: {accuracy:.4f}')
