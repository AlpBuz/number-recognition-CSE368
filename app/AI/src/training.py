import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import ImageFilter

# training the network
def train(network, optimizer, train_loader, epoch, log_interval, train_losses, train_counter, apply_blur=False, blur_radius=1.5):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    if apply_blur:
      # Apply blur to each image in the batch
      blurred_data = []
      for img in data:
          pil_image = to_pil_image(img)
          blurred_pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
          blurred_data.append(to_tensor(blurred_pil_image))
      data = torch.stack(blurred_data)


    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: (Blur: {}) {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(apply_blur,
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'app\AI\\results\model.pth')
      torch.save(optimizer.state_dict(), 'app\AI\\results\optimizer.pth')

# testing the network
def test(network, test_loader, test_losses, apply_blur=False, blur_radius=1.5):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      if apply_blur:
        #apply blur to each testing image
        blurred_data = []
        for img in data:
            pil_image = to_pil_image(img)
            blurred_pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))
            blurred_data.append(to_tensor(blurred_pil_image))  
        data = torch.stack(blurred_data)

      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
        
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))