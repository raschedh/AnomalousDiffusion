import torch.nn as nn
import torch.nn.functional as F
import torch

NUM_CLASSES = 4

class ClassificationModel(nn.Module):

    def __init__(self, num_inputs=10, num_hidden_units=64):
        super().__init__() 

        self.ln1 = nn.LayerNorm(num_inputs)
        self.lstm1 = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden_units, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.relu1 = nn.ReLU()

        self.ln2 = nn.LayerNorm(num_hidden_units * 2 + num_inputs)
        self.lstm2 = nn.LSTM(input_size=num_hidden_units * 2 + num_inputs, hidden_size=num_hidden_units, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.relu2 = nn.ReLU()

        self.ln3 = nn.LayerNorm(num_hidden_units*4 + num_inputs)
        self.lstm3 = nn.LSTM(input_size=num_hidden_units* 4 + num_inputs, hidden_size=num_hidden_units, num_layers=1, bidirectional=True, batch_first=True)
        self.relu3 = nn.ReLU()

        self.linear_layer = nn.Linear(num_hidden_units * 6 + num_inputs, NUM_CLASSES)

    def forward(self,x):

        x = self.ln1(x)

        skip1 = x 
        x, _ = self.lstm1(x)
        x = self.relu1(x)
        x = torch.cat((x, skip1), dim=-1)

        x = self.ln2(x)
        
        skip2 = x
        x, _ = self.lstm2(x)
        x = self.relu2(x)
        x = torch.cat((x, skip2), dim=-1)

        x = self.ln3(x)

        skip3 = x 
        x, _ = self.lstm3(x)
        x = self.relu3(x)
        out = torch.cat((x, skip3), dim=-1)
        
        out = self.linear_layer(out)
        out = F.log_softmax(out, dim=-1)

        return out
    

class RegressionModel(nn.Module):

    def __init__(self, num_inputs=10, num_hidden_units=64):
        super().__init__() 

        self.ln1 = nn.LayerNorm(num_inputs)
        self.lstm1 = nn.LSTM(input_size=num_inputs, hidden_size=num_hidden_units, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.relu1 = nn.ReLU()

        self.ln2 = nn.LayerNorm(num_hidden_units * 2 + num_inputs)
        self.lstm2 = nn.LSTM(input_size=num_hidden_units * 2 + num_inputs, hidden_size=num_hidden_units, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.relu2 = nn.ReLU()

        self.ln3 = nn.LayerNorm(num_hidden_units*4 + num_inputs)
        self.lstm3 = nn.LSTM(input_size=num_hidden_units* 4 + num_inputs, hidden_size=num_hidden_units, num_layers=1, bidirectional=True, batch_first=True)
        self.relu3 = nn.ReLU()

        self.linear_layer = nn.Linear(num_hidden_units * 6 + num_inputs, 1)

    def forward(self,x):

        x = self.ln1(x)

        skip1 = x 
        x, _ = self.lstm1(x)
        x = self.relu1(x)
        x = torch.cat((x, skip1), dim=-1)

        x = self.ln2(x)
        
        skip2 = x
        x, _ = self.lstm2(x)
        x = self.relu2(x)
        x = torch.cat((x, skip2), dim=-1)

        x = self.ln3(x)

        skip3 = x 
        x, _ = self.lstm3(x)
        x = self.relu3(x)
        out = torch.cat((x, skip3), dim=-1)
        
        out = self.linear_layer(out)

        return out

