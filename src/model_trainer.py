
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification




# Load the BERT model for sequence classification
def load_bert_model(num_labels, pretrained_model='bert-base-uncased'):
    model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
    return model




def train_model(model, train_inputs, train_labels, val_inputs, val_labels, num_epochs, batch_size, learning_rate):
    train_dataset = TensorDataset(torch.stack([inp['input_ids'] for inp in train_inputs]),
                                  torch.stack([inp['attention_mask'] for inp in train_inputs]),
                                  torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.stack([inp['input_ids'] for inp in val_inputs]),
                                torch.stack([inp['attention_mask'] for inp in val_inputs]),
                                torch.tensor(val_labels))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)  # Reduce LR on plateau

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {'input_ids': batch[0].squeeze(1).to(model.device),
                      'attention_mask': batch[1].squeeze(1).to(model.device),
                      'labels': batch[2].to(model.device)}

            input_ids, attention_mask, labels = inputs['input_ids'], inputs['attention_mask'], inputs['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Training Loss: {avg_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        for val_batch in val_loader:
            val_inputs = {'input_ids': val_batch[0].squeeze(1).to(model.device),
                          'attention_mask': val_batch[1].squeeze(1).to(model.device),
                          'labels': val_batch[2].to(model.device)}

            val_outputs = model(**val_inputs)
            val_loss += val_outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")

        model.train()
        scheduler.step(avg_val_loss)  # Adjust learning rate based on validation loss

    return model



# Evaluation function
def evaluate_model(model, test_inputs, test_labels):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inp in test_inputs:
            inputs = {'input_ids': torch.tensor(inp['input_ids'][0]).unsqueeze(0),
                      'attention_mask': torch.tensor(inp['attention_mask'][0]).unsqueeze(0)}

            # Check the shape of input_ids and attention_mask
            print(
                f"Input IDs Shape: {inputs['input_ids'].shape}, Attention Mask Shape: {inputs['attention_mask'].shape}")

            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred_label)

    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(test_labels, predictions, zero_division=1))






