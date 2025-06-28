import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.ffn(x)
    
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, use_gumbel=False, temperature=1.0):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_gumbel = use_gumbel
        self.temperature = temperature

    def forward(self, x):
        logits = self.gate(x)  # [B, E]

        if self.use_gumbel:
            # Gumbel-softmax sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            logits = (logits + gumbel_noise) / self.temperature

        weights = F.softmax(logits, dim=-1)  # [B, E]
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        return topk_weights, topk_indices, weights

class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, top_k=2, use_gumbel=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = GatingNetwork(input_dim, num_experts, top_k, use_gumbel)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        B, D =x.shape  #x:[B,D]
        topk_weights, topk_indices, all_weights = self.gate(x)
        output_dim = self.experts[0].ffn[-1].out_features
        final_output = torch.zeros(B, output_dim, device=x.device)

        expert_counts = torch.zeros(self.num_experts, device=x.device)

        for i in range(self.top_k):
            idx = topk_indices[:, i] #[B]
            w = topk_weights[:, i].unsqueeze(1) #[B,1]

            out_i = torch.stack([
                self.experts[e](x[j].unsqueeze(0)) for j,e in enumerate(idx)
            ]).squeeze(1)

            final_output += out_i*w
            expert_counts += torch.bincount(idx, minlength=self.num_experts)


        expert_prob = all_weights.mean(dim=0)
        load_balance_loss = (expert_prob * expert_prob).sum() * self.num_experts

        return final_output, load_balance_loss
    
model = MoE(input_dim=128, hidden_dim=64, output_dim=32, num_experts=4, top_k=2, use_gumbel=True) #.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

x = torch.randn(16,128)
y_true = torch.randn(16,32)

y_pred, lb_loss = model(x)
mse_loss = F.mse_loss(y_pred,y_true)

loss = mse_loss+0.01*lb_loss
loss.backward()
optimizer.step()