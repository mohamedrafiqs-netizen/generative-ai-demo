import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. GAN Components ---
class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# --- 2. VAE Components ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent_dim)
        self.log_var = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.mu(h), self.log_var(h)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# --- 3. Diffusion Model Utility ---
class DiffusionModel:
    def __init__(self, timesteps=1000):
        self.beta = torch.linspace(0.0001, 0.02, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])
        sqrt_one_minus = torch.sqrt(1. - self.alpha_bar[t])
        return sqrt_alpha_bar * x_0 + sqrt_one_minus * noise

# --- Execution Block ---
if __name__ == "__main__":
    print("--- Initializing Generative Models ---")
    
    # Test GAN
    gen = Generator(latent_dim=64, img_dim=784)
    disc = Discriminator(img_dim=784)
    z = torch.randn(5, 64)
    generated_data = gen(z)
    decision = disc(generated_data)
    print(f"GAN: Generated shape {generated_data.shape}, Discriminator output mean: {decision.mean().item():.4f}")

    # Test VAE
    vae = VAE(input_dim=784, latent_dim=20)
    fake_input = torch.randn(5, 784)
    recon, mu, log_var = vae(fake_input)
    print(f"VAE: Reconstruction shape {recon.shape}")

    # Test Diffusion
    dm = DiffusionModel()
    sample_input = torch.randn(5, 784)
    noisy_sample = dm.q_sample(sample_input, t=500)
    print(f"Diffusion: Noised data shape {noisy_sample.shape}")
    
    print("\nSUCCESS: All models initialized and executed forward pass.")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_my_data(data_path='./my_dataset', batch_size=32):
    # Define how to transform your images for the AI
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # If using B&W model
        transforms.Resize((28, 28)),                 # Resize to match model input
        transforms.ToTensor(),                       # Convert to AI numbers
        transforms.Normalize((0.5,), (0.5,))         # Normalize to range [-1, 1]
    ])

    # Load data from the folder
    # Note: ImageFolder requires the structure root/class/image.jpg
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
