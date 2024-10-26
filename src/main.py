import torch
import torchvision

# grab the right install here: https://pytorch.org/get-started/locally/
# source for below: https://nextjournal.com/gkoehler/pytorch-mnist
# Stopped at Building the Network 

EPOCHS = 5  # number of times we use the dataset

# training optimizations
LEARNING_RATE = 0.01
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

# training 
_ = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
        transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    batch_size=TRAINING_BATCH, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
        transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    batch_size=TESTING_BATCH, shuffle=True)



# plot 
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig.show()
plt.show()