import torch.nn as nn
import torch.nn.functional as F
import torch

class AlphaModel(nn.Module):

    def __init__(self, num_inputs, num_hidden_units=64):
        super().__init__() 

        self.ln1 = nn.LayerNorm(num_inputs)
        
        self.lstm1 = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden_units, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=num_hidden_units * 2 + num_inputs, hidden_size=num_hidden_units, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)

        self.ln2 = nn.LayerNorm(num_hidden_units*4 + num_inputs)
        self.lstm3 = nn.LSTM(input_size=num_hidden_units* 4 + num_inputs, hidden_size=num_hidden_units, num_layers=1, bidirectional=True, batch_first=True)

        self.linear_layer = nn.Linear(num_hidden_units * 6 + num_inputs, 1)

    def forward(self,x):

        x = self.ln1(x)

        skip1 = x 
        x, _ = self.lstm1(x)
        x = nn.ReLU()(x)
        x = torch.cat((x, skip1), dim=-1)
        
        skip2 = x
        x, _ = self.lstm2(x)
        x = nn.ReLU()(x)
        x = torch.cat((x, skip2), dim=-1)

        x = self.ln2(x)

        skip3 = x 
        x, _ = self.lstm3(x)
        x = nn.ReLU()(x)
        x = torch.cat((x, skip3), dim=-1)
        
        x = self.linear_layer(x)

        return x 


 