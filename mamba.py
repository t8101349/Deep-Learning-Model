import torch
import torch.nn as nn
import torch.nn.functional as F

class SSMKernel(nn.Module):
    def __init__(self, d_model, d_state, dt_rank):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.A = nn.Parameter(torch.randn(d_model, d_model))
        self.B = nn.Parameter(torch.randn(d_model, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_model))
        self.D = nn.Parameter(torch.randn(1, d_model)) #skip connection

        # gating/step update
        self.dt_project = nn.Linear(d_model, dt_rank)
        self.dt_project_out = nn.Linear(dt_rank, d_model)

    def forward(self, x):
        B, L, D = x.shape
        dt = self.dt_project_out(F.gelu(self.dt_project(x)))
        x_ssm = torch.zeros_like(x)

        '''
        for i in range(L):
            x_i = x[:,i,:] #[B,D]
            Ax = self.A * x_i #(B, D) * (D, d_state)x
            Bx = self.B * x_i
            Cx = self.C * x_i
            x_ssm[:, i, :] = (Ax + Bx + Cx) * dt[:, i, :] + self.D
        '''

        for i in range(L):
            x_i = x[:,i,:] #(B,D)
            Ax = torch.einsum('bd,ds->bs', x_i, self.A) # (B, D)*(B,D)
            Bx = torch.einsum('bd,ds->bs', x_i, self.B)
            Cx = torch.einsum('bd,ds->bs', x_i, self.C)
            out = (Ax + Bx + Cx)  # (B, D)
            out = out * dt[:,i,:] 
            x_ssm[:,i,:] = out + self.D
        # loop 寫法(低效) -> kernal寫法(高效) -> 使用官方模組    
        
        return x_ssm
    
class MambaBlock(nn.Module):
    def __init__(self, d_model=128, d_state=16, dt_rank=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SSMKernel(d_model, d_state, dt_rank)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-Norm SSM
        x_norm = self.norm1(x)
        ssm_out = self.ssm(x_norm)
        x = x + self.dropout(ssm_out)

        # Pre-Norm MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout(mlp_out)

        return x

    
    '''
    使用官方寫法
    from mamba_ssm import Mamba

    model = Mamba(d_model=128, d_state=16, expand=2, conv_kernel=15)
    output = model(input_tensor)  # input_tensor shape: (B, L, D)

    '''