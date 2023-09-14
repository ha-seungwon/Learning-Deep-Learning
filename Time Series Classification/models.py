import torch
import torch.nn as nn
import torch.nn.init as init

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.encoder_linear = nn.Linear(256, latent_dim * 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

        self.decoder_linear = nn.Linear(latent_dim*2, 256)

        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        print("x.shape", x.shape)
        # Encoding
        enc_output = self.encoder(x)
        print("enc_output.shape", enc_output.shape)
        enc_output = self.encoder_linear(enc_output)
        mu, logvar = torch.chunk(enc_output, 2, dim=1)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        print("z", z.shape)
        # Decoding
        z = self.decoder_linear(z)
        dec_output = self.decoder(z)
        print("dec_output.shape", dec_output.shape)
        return dec_output, z
