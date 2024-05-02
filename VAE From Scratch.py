import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

#-------------------------Build Model-------------------------#

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim = 200, z_dim = 20):
        super().__init__()
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma


    def decode(self,z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))


    def forward(self, x):
        mu, sigma = self.encode(x)
        e = torch.rand_like(sigma)
        z_parametrized = mu+sigma*e
        x_reconstucted = self.decode(z_parametrized)
        return x_reconstucted, mu, sigma




# if __name__ == '__main__':
#     x = torch.randn(4, 28*28)
#     vae = VariationalAutoEncoder(input_dim=28*28)
#     x_reconstucted, mu, sigma = vae(x)
#     print(x_reconstucted.shape)
#     print(mu.shape)
#     print(sigma.shape)

#-------------------------Key Params -------------------------#

device = torch.device('mps')
INPUT_DIM = 784
H_Dim = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 64
LR_RATE = 3e-3


#-------------------------Data Loader-------------------------#

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms. ToTensor(), download=True)
train_loader =DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

#-------------------------Model-------------------------#

Model = VariationalAutoEncoder(INPUT_DIM, H_Dim, Z_DIM).to(device)
optimizer = torch.optim.Adam(Model.parameters(), lr=LR_RATE)
loss_fun = nn.BCELoss(reduction='sum')

#-------------------------Trainning Loop-------------------------#

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x,_) in loop:
        # forward pass
        x = x.to(device).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = Model(x)

        # Compute loss
        r_loss = loss_fun(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        #Backprop
        loss = r_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())




















