import torch
from transformers import BertTokenizer
import pandas as pd


# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data
def tokenize_text(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )
    return inputs


# Load and tokenize data
def load_and_tokenize_data(data_path, save_path):
    try:
        tokenized_inputs = torch.load(save_path)
    except FileNotFoundError:
        data = pd.read_csv(data_path, sep='\t')
        tokenized_inputs = [tokenize_text(text) for text in data['review']]
        torch.save(tokenized_inputs, save_path)
    return tokenized_inputs

load_and_tokenize_data('../data/drugsComTrain_raw.tsv', '../data/train_inputs.pt')
load_and_tokenize_data('../data/drugsComTest_raw.tsv', '../data/test_inputs.pt')


# Convert rating labels to numerical form
def convert_to_numerical(data_path, save_path):
    data = pd.read_csv(data_path, sep='\t')
    labels = (data['rating'] - 1).astype(int).tolist()
    torch.save(labels, save_path)


convert_to_numerical('../data/drugsComTrain_raw.tsv', '../data/train_labels.pt')
convert_to_numerical('../data/drugsComTest_raw.tsv', '../data/test_labels.pt')

# Partition into classes for the 3-class problem
def partition_to_3class(rating):
    if rating <= 4:
        return 0
    elif 4 < rating < 7:
        return 1
    else:
        return 2

def convert_to_3class(data_path, save_path):
    data = pd.read_csv(data_path, sep='\t')
    labels = [partition_to_3class(rating) for rating in data['rating']]
    torch.save(labels, save_path)

convert_to_3class('../data/drugsComTrain_raw.tsv', '../data/train_labels_3class.pt')
convert_to_3class('../data/drugsComTest_raw.tsv', '../data/test_labels_3class.pt')


