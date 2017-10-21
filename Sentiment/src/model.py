import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_layer_size):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_layer_size)
        self.output = nn.Linear(hidden_layer_size, 2)

    def forward(self, inputs):
        x = self.hidden(inputs)
        x = F.tanh(x)
        x = self.output(x)
        x = F.log_softmax(x)
        return x