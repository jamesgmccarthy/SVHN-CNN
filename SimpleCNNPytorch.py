import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics as metrics

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(
    root='./cifardata', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(
    root='./cifardata', train=False, download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Training subset
n_training_samples = 20000
train_sampler = SubsetRandomSampler(
    np.arange(n_training_samples, dtype=np.int64))

# Validation subset
n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(
    n_training_samples, n_training_samples+n_val_samples, dtype=np.int64))

# Test subset
n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))


def output_size(in_size, kernel_size, stride, padding):
    output = int((in_size-kernel_size + 2*(padding))/stride) + 1
    return output


class SimpleCNN(torch.nn.Module):
    # our batch shape for input x is (channels, dim, dim) -> (3,32,32)
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # input channels =3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features
        self.fc1 = torch.nn.Linear(18*16*16, 64)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # computes the activation of the first convolution
        # size changes from (3,32,32) to (18,32,32)
        x = F.relu(self.conv1(x))

        # size changes from (18,32,32) to (18,16,16)
        x = self.pool(x)

        # Reshape data to the inpit layer of the nn
        # changes from (18,16,16) to (1,4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18*16*16)

        # Compute activation of the first fully connected layer
        # Size changes from (1,4608) to (1,64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected  layer (activation applied later)
        x = self.fc2(x)
        return x


def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    return train_loader


# Test and val loaders have constant batch sizes so can define them directly
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(
    train_set, batch_size=4, sampler=val_sampler, num_workers=2)


def create_loss_optim(net, lr=0.0001):
    # loss function
    loss = torch.nn.CrossEntropyLoss()

    # optimizer
    opt = optim.Adam(net.parameters(), lr=lr)

    return (loss, opt)


def train_net(net, batch_size, n_epochs, learning_rate):
    # Print all the hyperparams of training iter
    print("====HYPERPARAMETERS====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("="*26)

    # Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss, optimizer = create_loss_optim(net, learning_rate)

    # Time for pretraining
    training_start_time = time.time()

    # Loop for n_epochs
    net.train()
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches//10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            # Get inputs
            inputs, labels = data

            # Wrap them in Variable object
            inputs, labels = Variable(inputs), Variable(labels)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Print Stats

            running_loss += loss_size.item()
            total_train_loss += loss_size.item()

            # Print every 10th batch of an epoch
            if (i+1) % (print_every+1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took {:.2f}".format(
                    epoch+1, int(100*(i+1)/n_batches), running_loss/print_every, time.time()-start_time))
                # reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                # wrap tensors in variables
                inputs, labels = Variable(inputs), Variable(labels)

                # Forward pass
                val_outputs = net(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.item()

            print(f"Validation loss = {total_val_loss/len(val_loader)}")

    print(f"Training finished, took {time.time() - training_start_time}")


def calculate_metrics(net):
    net.eval()
    epoch_loss = 0
    outputs = []
    acc = 0
    f1 = 0
    recall = 0
    precision = 0
    epoch = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            epoch += 1
            inputs, labels_var = Variable(inputs), Variable(labels)
            output = net(inputs)
            acc += metrics.accuracy_score(output.argmax(1), labels)
            f1 += metrics.f1_score(output.argmax(1), labels)
            recall += metrics.recall_score(output.argmax(1), labels)
            precision += metrics.precision_score(output.argmax(1), labels)
    acc = acc/epoch
    f1 = f1/epoch
    recall = recall / epoch
    precision = precision/epoch
    print("Acc:", acc)
    print("F1:", f1)
    print("recall:", recall)
    print("Precision:", precision)


net = SimpleCNN()
train_net(net, 128, 10, 0.001)
