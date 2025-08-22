import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 64
batch_size = 128
epochs = 100
lr = 0.0002

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # scale to [-1, 1]
])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(-1, 784))

# Initialize models
G = Generator()
D = Discriminator()

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# Training
for epoch in range(epochs):
    for real_imgs, _ in loader:
        batch_size = real_imgs.size(0)

        # Labels
        real = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = G(z)

        D_real = D(real_imgs)
        D_fake = D(fake_imgs.detach())

        loss_D = criterion(D_real, real) + criterion(D_fake, fake)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        D_fake = D(fake_imgs)
        loss_G = criterion(D_fake, real)  # wants discriminator to think fake is real
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{epochs}  Loss D: {loss_D.item():.4f}  Loss G: {loss_G.item():.4f}")

    # Show samples
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            sample = G(torch.randn(16, latent_dim))
            grid = sample.view(16, 28, 28).numpy()
            fig, axes = plt.subplots(4, 4, figsize=(4, 4))
            for i, ax in enumerate(axes.flat):
                ax.imshow(grid[i], cmap='gray')
                ax.axis('off')
            plt.show()
