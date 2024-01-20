# pip install torch transformers
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Sample text classification dataset
data = {
    'text': ['This is a positive example.', 'Negative sentiment here.', 'Another positive statement.'],
    'label': [1, 0, 1]
}

df = pd.DataFrame(data)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                            num_labels=2)  # Adjust 'num_labels' based on your classification task

# Tokenize and encode the training set
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

tokenized_train = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
train_dataset = TensorDataset(tokenized_train['input_ids'], tokenized_train['attention_mask'],
                              torch.tensor(train_labels))

# Tokenize and encode the testing set
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

tokenized_test = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')
test_dataset = TensorDataset(tokenized_test['input_ids'], tokenized_test['attention_mask'], torch.tensor(test_labels))

# Create DataLoader for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Training loop (example, you might need to customize it based on your specific needs)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(3):  # Replace 3 with the desired number of epochs
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss}")

# Evaluate on the test set
model.eval()
correct_predictions = 0

for batch in tqdm(test_loader, desc='Evaluating'):
    input_ids, attention_mask, labels = batch
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    predictions = torch.argmax(outputs.logits, dim=1)
    correct_predictions += (predictions == labels).sum().item()

accuracy = correct_predictions / len(test_loader.dataset)
print(f"Accuracy on the test set: {accuracy:.2%}")
