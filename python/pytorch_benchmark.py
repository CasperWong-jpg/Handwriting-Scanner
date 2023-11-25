import multiprocessing
import torch
import ssl
import scipy
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from nn import get_random_batches
from pytoch_nn import *

# Need to turn off SSL verification to load datasets on Mac
num_workers = multiprocessing.cpu_count()
ssl._create_default_https_context = ssl._create_unverified_context

# Train neural network on MNIST36
def nn_MNIST36():
    batch_size = 108
    max_iters = 80

    # Load and normalize MNIST36 dataset
    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
    train_x, train_y = train_data['train_data'], train_data['train_labels']
    valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
    batches = get_random_batches(train_x, train_y, batch_size)
    num_batches = len(batches)

    # Define NN, loss function, optimzer
    lr = 0.005
    net = NeuralNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # Train MNIST36 network
    epoch_list = []
    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []

    for epoch in range(max_iters):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        for inputs, labels in batches:
            inputs, labels = torch.from_numpy(inputs).to(torch.float32), torch.from_numpy(labels).to(torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            num_correct = torch.sum(torch.argmax(outputs, axis=1) == torch.argmax(labels, axis=1)).item()
            running_acc += num_correct / batch_size

        if epoch % 2 == 1:
            # Run on validation set too
            valid_inputs = torch.from_numpy(valid_x).to(torch.float32)
            valid_labels = torch.from_numpy(valid_y).to(torch.float32)
            outputs = net(valid_inputs)
            loss = criterion(outputs, valid_labels).item()
            num_correct = torch.sum(torch.argmax(outputs, axis=1) == torch.argmax(valid_labels, axis=1)).item()
            acc = num_correct / valid_y.shape[0]

            # Report results
            epoch_list.append(epoch)
            train_acc_list.append(running_acc / num_batches)
            train_loss_list.append(running_loss / num_batches)
            valid_acc_list.append(acc)
            valid_loss_list.append(loss)

    fig, (acc_plt, loss_plt) = plt.subplots(2, 1)
    fig.suptitle(f"Learning rate = {lr}. Batch size = {batch_size}")
    acc_plt.plot(epoch_list, train_acc_list, label="Training accuracy")
    acc_plt.plot(epoch_list, valid_acc_list, label="Validation accuracy")
    acc_plt.set_ylabel("Accuracy")
    acc_plt.legend()
    loss_plt.plot(epoch_list, train_loss_list, label="Training loss")
    loss_plt.plot(epoch_list, valid_loss_list, label="Validation loss")
    loss_plt.set_ylabel("Loss")
    loss_plt.legend()
    plt.show()


# Train convolutional neural network on MNIST36
def cnn_MNIST36():
    batch_size = 108
    max_iters = 50

    # Load and normalize MNIST36 dataset
    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
    train_x, train_y = train_data['train_data'], train_data['train_labels']
    valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
    batches = get_random_batches(train_x, train_y, batch_size)
    num_batches = len(batches)

    # Define NN, loss function, optimzer
    lr = 0.005
    net = ConvNetComplex(numChannels=1, numOutputs=36)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # Train MNIST36 network
    epoch_list = []
    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []

    for epoch in range(max_iters):  # loop over the dataset multiple times
        print(f"Training epoch: {epoch}")
        running_loss = 0.0
        running_acc = 0.0
        for inputs, labels in batches:
            inputs = torch.reshape(torch.from_numpy(inputs).to(torch.float32), (batch_size, 1, 32, 32))
            labels = torch.from_numpy(labels).to(torch.float32)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            num_correct = torch.sum(torch.argmax(outputs, axis=1) == torch.argmax(labels, axis=1)).item()
            running_acc += num_correct / batch_size

        if epoch % 2 == 1:
            # Run on validation set too
            valid_inputs = torch.reshape(torch.from_numpy(valid_x).to(torch.float32), (valid_x.shape[0], 1, 32, 32))
            valid_labels = torch.from_numpy(valid_y).to(torch.float32)
            outputs = net(valid_inputs)
            loss = criterion(outputs, valid_labels).item()
            num_correct = torch.sum(torch.argmax(outputs, axis=1) == torch.argmax(valid_labels, axis=1)).item()
            acc = num_correct / valid_y.shape[0]

            # Report results
            epoch_list.append(epoch)
            train_acc_list.append(running_acc / num_batches)
            train_loss_list.append(running_loss / num_batches)
            valid_acc_list.append(acc)
            valid_loss_list.append(loss)

    fig, (acc_plt, loss_plt) = plt.subplots(2, 1)
    fig.suptitle(f"Learning rate = {lr}. Batch size = {batch_size}")
    acc_plt.plot(epoch_list, train_acc_list, label="Training accuracy")
    acc_plt.plot(epoch_list, valid_acc_list, label="Validation accuracy")
    acc_plt.set_ylabel("Accuracy")
    acc_plt.legend()
    loss_plt.plot(epoch_list, train_loss_list, label="Training loss")
    loss_plt.plot(epoch_list, valid_loss_list, label="Validation loss")
    loss_plt.set_ylabel("Loss")
    loss_plt.legend()
    plt.show()


# Train CNN on CIFAR10 also
def cnn_CIFAR10():
    batch_size = 250
    max_iters = 25

    # Load and normalize CIFAR10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                              num_workers=2, persistent_workers=True)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=2, persistent_workers=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Define CNN
    net = ConvNet()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # Train CIFAR10 CNN
    epoch_list = []
    train_acc_list = []
    train_loss_list = []
    for epoch in range(max_iters):  # loop over the dataset multiple times
        print(f"Training epoch: {epoch}")
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Record statistics
            running_loss += loss.item()
            num_correct = torch.sum(torch.argmax(outputs, axis=1) == labels).item()
            running_acc += num_correct / batch_size

        # Report results
        epoch_list.append(epoch)
        train_acc_list.append(running_acc / (i + 1))
        train_loss_list.append(running_loss / (i + 1))

    fig, (acc_plt, loss_plt) = plt.subplots(2, 1)
    fig.suptitle(f"Learning rate = {lr}. Batch size = {batch_size}")
    acc_plt.plot(epoch_list, train_acc_list, label="Training accuracy")
    acc_plt.set_ylabel("Accuracy")
    acc_plt.legend()
    loss_plt.plot(epoch_list, train_loss_list, label="Training loss")
    loss_plt.set_ylabel("Loss")
    loss_plt.legend()
    plt.show()

    # Test CIFAR10 CNN
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

if __name__ == '__main__':
    nn_MNIST36()  # Takes a few seconds to run
    cnn_MNIST36()  # Takes a minute to run
    # cnn_CIFAR10()  # Takes a couple of minutes to run