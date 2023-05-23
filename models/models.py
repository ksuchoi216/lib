import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out




def get_loss_fn(loss_name='cross_ent'):
    match loss_name:
        case 'cross_ent':
            loss_fn = nn.CrossEntropyLoss()

    return loss_fn

def get_optimizer(model, opt_name = 'Adam', lr=0.005):    
    match opt_name:
        case 'NAdam':
            optimizer = torch.optim.NAdam(
                model.parameters(), lr=lr)
        case 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr)

    return optimizer
