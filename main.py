import torch
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from data_load import load_data
from token_to_vector import encode_text_data
from model import BayesianNeuralNetworkWithBN

# data load
data_directory = 'data'
emails, labels = load_data(data_directory)

X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# text to vector, using Bert
encoded_train_data = encode_text_data(X_train, max_length=32)
encoded_test_data = encode_text_data(X_test, max_length=32)

# prepare the loader
train_da = torch.from_numpy(encoded_train_data).view(encoded_train_data.shape[0], -1)
test_da = torch.from_numpy(encoded_test_data).view(encoded_test_data.shape[0], -1)

batch_size = 32

train_dataset = torch.utils.data.TensorDataset(train_da, torch.tensor(y_train))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(test_da, torch.tensor(y_test))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Hyperparameters
input_size = train_da.shape[1]
hidden_size = 64
output_size = 2  
learning_rate = 0.01
num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BayesianNeuralNetworkWithBN(input_size, hidden_size, output_size, dropout_rate=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def compute_kl_div(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# train and evaluation
for epoch in range(num_epochs):
    total_loss = 0.0
    model.train() 

    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        outputs, params = model(batch_inputs)

        weights_in_hidden, log_var_weights_in_hidden, weights_hidden_out, log_var_weights_hidden_out, bias_hidden, log_var_bias_hidden, bias_out, log_var_bias_out = params

        nll_loss = F.cross_entropy(outputs, batch_labels)

        kl_loss = compute_kl_div(weights_in_hidden, log_var_weights_in_hidden) + \
                  compute_kl_div(weights_hidden_out, log_var_weights_hidden_out) + \
                  compute_kl_div(bias_hidden, log_var_bias_hidden) + \
                  compute_kl_div(bias_out, log_var_bias_out)

        loss = nll_loss + 1e-4 * kl_loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)


    model.eval()
    with torch.no_grad():
        test_len = 0
        right = 0
        for batch_test, batch_test_label in test_loader:
            batch_test = batch_test.to(device)
            batch_test_label = batch_test_label.to(device)

            test_outputs, _ = model(batch_test)
            predictions = torch.argmax(test_outputs, dim=1)

            right += torch.sum(predictions == batch_test_label).item()
            test_len += batch_test_label.shape[0]

    accuracy = right/test_len

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss}  ', "Accuracy on test data:", accuracy)
