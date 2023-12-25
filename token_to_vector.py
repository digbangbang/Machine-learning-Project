import torch
import numpy as np
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_text_data(text_data, tokenizer = tokenizer, model = model, max_length=32):
    encoded_data = []

    for text in text_data:
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)

        representation = outputs.last_hidden_state[:, 0, :].numpy()
        encoded_data.append(representation)

    return np.array(encoded_data)