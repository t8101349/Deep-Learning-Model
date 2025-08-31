import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --------------------------------------------------
# 1️⃣ Patch Embedding
# --------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, input_dim=3, embed_dim=96, patch_size=4):
        super.__init__()
        self.proj = nn. Conv2d(input_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self,x):
        x = self.proj(x)
        x = x.rearrange(x, 'b c h w -> b (h w) c')
        return x

# --------------------------------------------------
# 2️⃣ Window Partition / Reverse
# --------------------------------------------------
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B,H//window_size,window_size,W//window_size,window_size,C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size*window_size,C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0]/ (H*W/window_size/window_size))
    x = windows.view(B, H//window_size, W//window_size, -1)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(B,H,W,-1)
    return x

# --------------------------------------------------
# 3️⃣ Window-based Multi-Head Self-Attention
# --------------------------------------------------
class WindowAttemtion(nn.Module):
    def __init__(self, embed_dim, window_size, num_heads):
        super().__init__()
        self.dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim*embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim,embed_dim)

        self.scale = (embed_dim//num_heads) **-0.5

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, C//heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out

# --------------------------------------------------
# 4️⃣ Swin Transformer Block
# --------------------------------------------------
class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution,num_heads, window_size = 7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size= shift_size

        self.norm1 = nn.LinearNorm(dim)
        self.attn = WindowAttemtion(dim, window_size, num_heads)

        self.norm2 = nn.LinearNorm(dim)
        self.hidden_dim = int(dim*mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim,self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim),
        )

    def forward(self, x):
        H , W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shift=(-self.shift_size, -self.shift_size), dims=(1,2))
        else:
            shifted_x = x
        
        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows =self.attn(self.norm1(x_windows))

        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shift=(self.shift_size, self.shift_size), dims=(1,2))

        x = x.view(B, H*W , C)
        x = x + self.mlp(self.norm2(x)) 
        return x
    
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm =nn.LayerNorm(4*dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        X0 = x[:, 0::2, 0::2, :]
        X1 = x[:, 1::2, 0::2, :]
        X2 = x[:, 0::2, 1::2, :]
        X3 = x[:, 1::2, 1::2, :]
        x = torch.cat([X0, X1, X2, X3],-1)
        x = x.view(B, -1, 4*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x
    
class SwinStage(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if 1%2 == 0 else window_size//2
            self.blocks.append(
                SwinBlock(dim, input_resolution, num_heads, window_size, shift_size)
                # W-MSA, SW-MSA交錯
            )
        
        def forward(self,x):
            for block in self.blocks:
                x = block(x)
            return x
        
class SwinTransformer(nn.Module):
    def __init__(self, image_size =224, patch_size=4, in_channels=3,
                 embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=7):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.pos_drop = nn.Dropout(p=0.0)

        self.stages = nn.ModuleList()
        self.patches_resolution = (image_size//patch_size, image_size//patch_size)
        dim = embed_dim

        for i in range(len(depths)):
            stage = SwinStage(dim, self.patches_resolution, depths[i], num_heads[i], window_size)
            self.stages.append(stage)

            if i < len(depths) - 1:
                self.stages.append(PatchMerging(self.patches_resolution, dim))
                self.patches_resolution = (self.patches_resolution[0]//2, self.patches_resolution[1]//2)
                dim *= 2
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        return x  # [B,L,C]