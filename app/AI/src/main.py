import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
# for continuing training
import os
# functions/classes
from network import Net
from training import train, test
import matplotlib.pyplot as plt


# grab the right install here: https://pytorch.org/get-started/locally/
# source for below: https://nextjournal.com/gkoehler/pytorch-mnist
# Stopped at Building the Network 

''' 
MODEL IS CURRENTLY TRAINED ON 10 EPOCHS
Test set: Avg. loss: 0.0786, Accuracy: 9753/10000 (98%) after 10 epochs
to continue training:
run 'python3 main.py'
enter the last completed epoch to continue from (curently 10) to better the model
'''

EPOCHS = 10  # number of times we use the dataset

# training optimizations
LEARNING_RATE = 0.05
MOMENTUM = 0.5

# batch sizes
TRAINING_BATCH = 100
TESTING_BATCH = 1000

# for repeatable results
seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(seed)

# more datasets can be found in: https://pytorch.org/vision/stable/datasets.html
# MNIST chosen by tutorial

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor(),
    download = True
)


# loading training data 
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download=True,
        transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    batch_size=TRAINING_BATCH, shuffle=True)

# lodaing testing data
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=False, download=True,
        transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    batch_size=TESTING_BATCH, shuffle=True)

# creating the network
learning_rate = 0.05   
momentum = 0.5
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# training logs
train_losses = []
train_counter = []
test_losses = []
train_accs = []
train_accs_blur = []
test_accs = []
test_accs_blur = []
log_interval = 10 # determines how frequently progress is printed, 100 would be less frequent

# checking if we have a saved model to continue training, if not we start from scratch
start_epoch = 1 # default epoch to start from
if os.path.exists('app\AI\\results\model.pth') and os.path.exists('app\AI\\results\optimizer.pth'):
    print("Model already found, continuing training.")
    network.load_state_dict(torch.load('app\AI\\results\model.pth'))
    optimizer.load_state_dict(torch.load('app\AI\\results\optimizer.pth'))
    start_epoch = int(input("Enter the last completed epoch to continue from: ")) + 1
else:
    print("No model found, starting from scratch.")

test(network, test_loader, test_losses)
for epoch in range(start_epoch, EPOCHS + start_epoch):
    train_accs.append(train(network, optimizer, train_loader, epoch, log_interval, train_losses, train_counter))
    train_accs_blur.append(train(network, optimizer, train_loader, epoch, log_interval, train_losses, train_counter, apply_blur=True, blur_radius=2.0))
    test_accs.append(test(network, test_loader, test_losses))
    test_accs_blur.append(test(network, test_loader, test_losses, apply_blur=True, blur_radius=2.0))

trainClear = plt.plot([e for e in range(len(train_accs))], train_accs, label="Train")
testClear = plt.plot([e for e in range(len(test_accs))], test_accs, label="Test")
trainBlur = plt.plot([e for e in range(len(train_accs_blur))], train_accs_blur, label="Train (Blur)")
testBlur = plt.plot([e for e in range(len(test_accs_blur))], test_accs_blur, label="Test (Blur)")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Credit to Trenton McKinney at https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend for preventing duplicate labels