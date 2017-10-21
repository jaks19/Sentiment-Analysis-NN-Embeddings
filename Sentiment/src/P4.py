# My library of functions
import train_and_evaluate as trainer

# Torch libraries
import torch.nn as nn
import torch.nn.functional as F

# Tunable Params for Neural Net
input_size = 300
hidden_layer_size = input_size * 2
learning_rate = 0.1
weight_decay_constant = 10**-5
num_epochs = 50

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

if __name__ == '__main__':
    trainer.train_and_evaluate('test')