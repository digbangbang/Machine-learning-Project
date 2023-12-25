import torch
import torch.nn as nn


class BayesianNeuralNetworkWithBN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(BayesianNeuralNetworkWithBN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_in_hidden_mu = nn.Parameter(torch.randn(input_size, hidden_size))
        self.weights_in_hidden_log_var = nn.Parameter(torch.randn(input_size, hidden_size))
        self.weights_hidden_out_mu = nn.Parameter(torch.randn(hidden_size, output_size))
        self.weights_hidden_out_log_var = nn.Parameter(torch.randn(hidden_size, output_size))
        self.bias_hidden_mu = nn.Parameter(torch.randn(hidden_size))
        self.bias_hidden_log_var = nn.Parameter(torch.randn(hidden_size))
        self.bias_out_mu = nn.Parameter(torch.randn(output_size))
        self.bias_out_log_var = nn.Parameter(torch.randn(output_size))

        self.bn_input = nn.BatchNorm1d(input_size)
        self.bn_hidden = nn.BatchNorm1d(hidden_size)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.bn_input(x)

        weights_in_hidden = self.weights_in_hidden_mu + torch.exp(0.5 * self.weights_in_hidden_log_var) * torch.randn_like(self.weights_in_hidden_log_var)
        weights_hidden_out = self.weights_hidden_out_mu + torch.exp(0.5 * self.weights_hidden_out_log_var) * torch.randn_like(self.weights_hidden_out_log_var)
        bias_hidden = self.bias_hidden_mu + torch.exp(0.5 * self.bias_hidden_log_var) * torch.randn_like(self.bias_hidden_log_var)
        bias_out = self.bias_out_mu + torch.exp(0.5 * self.bias_out_log_var) * torch.randn_like(self.bias_out_log_var)

        hidden = torch.tanh(torch.matmul(x, weights_in_hidden) + bias_hidden)

        hidden = self.bn_hidden(hidden)

        hidden = self.dropout(hidden)

        output = torch.matmul(hidden, weights_hidden_out) + bias_out
        return output, (weights_in_hidden, self.weights_in_hidden_log_var,
                        weights_hidden_out, self.weights_hidden_out_log_var,
                        bias_hidden, self.bias_hidden_log_var,
                        bias_out, self.bias_out_log_var)

