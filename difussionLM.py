# Minimal runnable version of Diffusion-LM
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified Transformer Block
def get_transformer_block(embed_dim, hidden_dim):
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim),
        num_layers=2
    )

class DiffusionLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, T=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.time_embed = nn.Embedding(T, embed_dim)
        self.denoiser = get_transformer_block(embed_dim, hidden_dim)

        # Diffusion schedule
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def q_sample(self, z_0, t, noise):
        # z_0: (B, L, D), t: (B,), noise: (B, L, D)
        B, L, D = z_0.shape
        alpha_bar = self.alpha_bars[t].view(B, 1, 1)  # (B, 1, 1)
        return torch.sqrt(alpha_bar) * z_0 + torch.sqrt(1 - alpha_bar) * noise

    def forward(self, x_tokens, t):
        B, L = x_tokens.shape
        z_0 = self.embedding(x_tokens)  # (B, L, D)
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)

        # Add time embedding
        t_embed = self.time_embed(t).unsqueeze(1)  # (B, 1, D)
        z_input = z_t + t_embed

        z_input = z_input.transpose(0, 1)  # (L, B, D)
        z_out = self.denoiser(z_input).transpose(0, 1)  # (B, L, D)
        return z_out, noise  # predicted, ground truth

# Run test
if __name__ == "__main__":
    torch.manual_seed(42)
    model = DiffusionLM(vocab_size=1000, embed_dim=64, hidden_dim=128, T=100)

    x_tokens = torch.randint(0, 1000, (4, 16))  # (B=4, L=16)
    t = torch.randint(0, 100, (4,))

    pred_noise, true_noise = model(x_tokens, t)
    loss = F.mse_loss(pred_noise, true_noise)
    print("MSE Loss:", loss.item())
