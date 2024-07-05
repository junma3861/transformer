import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

# Sample Dataset
class SampleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten()
        }

# Sample Text Data
texts = ["This is a sample text.", "Another example sentence.", "More data for pre-training."]

# Parameters
MAX_LEN = 20
BATCH_SIZE = 2
EPOCHS = 2
LR = 1e-5

# Tokenizer and Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SampleDataset(texts, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model Definition
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.linear(sequence_output)
        return logits

# Initialize Model, Optimizer, and Loss Function
model = TransformerModel()
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training Loop
model.train()
for epoch in range(EPOCHS):
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)

        # Shift the logits and labels to align them for prediction
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Evaluation
model.eval()
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = model(input_ids, attention_mask)
        print(outputs)

print("Pre-training complete.")
