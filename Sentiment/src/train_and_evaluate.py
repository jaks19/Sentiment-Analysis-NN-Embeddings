# My library of functions
from model import Net
import data
import matplotlib.pyplot as plt


# Torch libraries
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


# Tunable Params for Neural Net
input_size = 300
hidden_layer_size = input_size * 2
learning_rate = 0.1
weight_decay_constant = 10**-5
num_epochs = 50


# One epoch within the training cycle

def run_one_epoch(model, train_batches, eval_data, loss_function, optimizer):
    for batch in train_batches:
        model.train()
        x = Variable(torch.FloatTensor(batch[0]))
        y = Variable(torch.LongTensor(batch[1]))
        output = model(x)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    x = Variable(torch.FloatTensor(eval_data[0]))
    y = eval_data[1]
    eval_classifications = model(x)
    eval_accuracy_percentage = eval_loss(eval_classifications, y)

    return eval_accuracy_percentage


# Helper to count accuracy percentage

def eval_loss(generated_classifications, true_classifications):
    assert len(generated_classifications) == len(true_classifications)
    correct = 0
    for i, probability_pair in enumerate(generated_classifications):
        label = 0
        np_probability_pair = probability_pair.data.numpy()
        if np_probability_pair[1] > np_probability_pair[0]: label = 1
        if label == true_classifications[i]: correct += 1
    return (correct * 1.0 / len(generated_classifications)) * 100


# Runs the Neural net training and evaluates on chosen data set (dev or test)

def train_and_evaluate(evaluation_set):
    all_data = data.prepare_all_data()
    train_data = all_data['train']
    train_batches = data.get_batches(train_data[0], train_data[1])
    eval_data = all_data[evaluation_set]

    model = Net(input_size, hidden_layer_size)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay_constant)

    eval_accuracies = []
    for i in range(num_epochs):
        eval_accuracy_percentage = run_one_epoch(model, train_batches, eval_data, loss_function, optimizer)
        eval_accuracies.append(eval_accuracy_percentage)

    final_accuracy = eval_accuracies[-1]
    best_accuracy = max(eval_accuracies)
    best_epoch = eval_accuracies.index(best_accuracy)

    print 'Hidden layer size: ', hidden_layer_size*1.0/input_size
    print 'Learning rate: ', learning_rate
    print 'Weight decay constant: ', weight_decay_constant

    print 'The FINAL accuracy using this model on the ', evaluation_set, ' set is: ', final_accuracy
    print 'The BEST accuracy using this model on the ', evaluation_set, ' set is: ', best_accuracy
    print 'BEST accuracy occurs at epoch: ', best_epoch
    print ' '

    return


# Runs across the range of Hyper-Parameters

def tune():
    global hidden_layer_size
    global learning_rate
    global weight_decay_constant

    for hidden in [input_size/4, input_size/2, input_size, 2*input_size]:
        for lr in [10**-5, 10**-3, 10**-1, 10**1]:
            for wdc in [10**-5, 10**-3, 10**1]:

                hidden_layer_size = hidden
                learning_rate = lr
                weight_decay_constant = wdc

                train_and_evaluate('dev')

    return


# Runs across domain of one hyper parameter to get performance vs parameter values

def plot(x, performances, name, xlabel):
    plt.clf()
    plt.suptitle(name)
    plt.plot(x, performances, 'b-', marker='o')
    plt.xlabel(xlabel)
    plt.ylabel('Dev Performance')
    plt.savefig(name)
    return


def reset_params_to_best():
    global hidden_layer_size
    global learning_rate
    global weight_decay_constant

    hidden_layer_size = 2 * input_size
    learning_rate = 0.1
    weight_decay_constant = 10 ** -5
    return


def vary_params_and_plot():
    global hidden_layer_size
    global learning_rate
    global weight_decay_constant

    reset_params_to_best()
    performances = []
    domain = [input_size/5, input_size/4, input_size/3, input_size/2, input_size, 2*input_size, 4*input_size, 8*input_size, 12*input_size]
    for hidden in domain:
        print hidden
        hidden_layer_size = hidden
        performances.append(train_and_evaluate('dev')[2])

    plot(domain, performances, 'Performance vs Hidden Layer Size', 'Hidden layer size')

    reset_params_to_best()
    performances = []
    domain = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10, 100, 1000]
    for lr in domain:
        learning_rate = lr
        performances.append(train_and_evaluate('dev')[2])

    plot(domain, performances, 'Performance vs Learning Rate', 'Learning Rate')

    reset_params_to_best()
    performances = []
    domain = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for wdc in domain:
        weight_decay_constant = wdc
        performances.append(train_and_evaluate('dev')[2])

    plot(domain, performances, 'Performance vs Weight Decay Constant', 'Weight Decay Constant')

    return