import torch
import torch.nn as nn
import torch.optim as optim

size = 128

class PayloadModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = nn.Sequential( \
      *[nn.Conv1d(size, size, 1, 1, 0) for i in range(10)])
  
  def forward(self, X):
    return self.layers(X)

device = "cuda:0"

model = PayloadModel().to(device)
optimizer = optim.Adam(model.parameters(), lr = 5e-4)

inputs  = torch.randn(32, size, 256, device = device)
targets = torch.randn(32, size, 256, device = device)
loss = nn.MSELoss()

for step in range(10000000):
  predicted = model(inputs)
  L = loss(predicted, targets)
  
  optimizer.zero_grad()
  L.backward()
  optimizer.step()

  print("%d steps, loss = %f" % (step, L.item()), end = "\r")
