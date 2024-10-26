import torch
from torch import nn
import ipdb
from einops import rearrange

#import torch_xla
#import torch_xla.core.xla_model as xm

#device = xm.xla_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):   # dim=512, hidden_dim(mlp_dim)=512
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)  # (128, 200, 512)

class Attention(nn.Module):     # dim=512, depth=2, heads=4, dim_head=128,
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 128 * 4 = 512
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(2*inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):                           # input             (128, 200, 512)
        b,n,d=x.size()
        qkvt = self.to_qkv(x).chunk(4, dim = -1)    # each chunk        (128, 200, 512)
        q, k, v, t = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkvt)  # (128, 4, 200, 128)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # root(d_k) (128, 4, 200, 200)

        attn1 = self.attend(dots)                   #                   (128, 4, 200, 200)

        #tmp_ones = torch.ones(n).cuda()             #                   (200, )
        #tmp_n = torch.linspace(1, n, n).cuda()      #                   (200, )
        tmp_ones = torch.ones(n).to(device)
        tmp_n = torch.linspace(1, n, n).to(device)
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1,1)) #       (200, 200)
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.)))    #   (200, 200)
        attn2 = (attn2 / attn2.sum(-1)).unsqueeze(0).unsqueeze(1).repeat(b,self.heads, 1, 1)    # (128, 4, 200, 200)

        out = torch.cat([torch.matmul(attn1, v),torch.matmul(attn2, t)],dim=-1) # (128, 4, 200, 256)
        out = rearrange(out, 'b h n d -> b n (h d)')    # (128, 200, 1024)
        return self.to_out(out)                         # (128, 200, 512)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.): # dim=512, depth=2, heads=4, dim_head=128,
        super().__init__()                                                  # mlp_dim=512
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):               # input             (128, 200, 512)
        for attn, ff in self.layers:    # depth=2, loop 2 times
            x = attn(x) + x             #                   (128, 200, 512)
            x = ff(x) + x               #                   (128, 200, 512)
        return x                        #                   (128, 200, 512)