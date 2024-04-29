import torch
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report


# Evaluation function
def evaluate_model(model, test_inputs, test_labels):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inp in test_inputs:
            inputs = {'input_ids': torch.tensor(inp['input_ids'][0]).unsqueeze(0),
                      'attention_mask': torch.tensor(inp['attention_mask'][0]).unsqueeze(0)}


            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred_label)

    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(test_labels, predictions, zero_division=1))





test_inputs = torch.load("../data/test_inputs.pt")
test_labels = torch.load("../data/test_labels.pt")

# Load a smaller subset of the test set
subset_size = 1000
test_inputs_subset = test_inputs[:subset_size]
test_labels_subset = test_labels[:subset_size]


# Load the BERT model for the 10-class problem
model_path = "../models/model_10class_subset_10000.pth"
num_labels = 10  # Assuming there are 10 classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model_state_dict = torch.load(model_path)
model.load_state_dict(model_state_dict)

# Use your evaluate_model function on the subset
evaluate_model(model, test_inputs_subset, test_labels_subset)


test_inputs = torch.load("../data/test_inputs.pt")
test_labels = torch.load("../data/test_labels_3class.pt")

# Load a smaller subset of the test set
subset_size = 1000
test_inputs_subset = test_inputs[:subset_size]
test_labels_subset = test_labels[:subset_size]

# Load the BERT model for the 3-class problem
model_path = "../models/model_3class_subset_10000.pth"
num_labels = 3  # Assuming there are 3 classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model_state_dict = torch.load(model_path)
model.load_state_dict(model_state_dict)

# Use your evaluate_model function on the subset
evaluate_model(model, test_inputs_subset, test_labels_subset)
