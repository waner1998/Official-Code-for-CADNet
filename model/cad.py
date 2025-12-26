import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from untils.sto_depth import DropPath
# from main import args
import time
def conv_3x3_bn(inp, oup, image_size, depth, channel_out, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class LayerScale(nn.Module):
    def __init__(self, dim, fn, downsample, depth):
        super().__init__()
        if downsample:
            dim = int(dim * 2)
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn
        # print(dim)
    def forward(self, x, **kwargs):
        # print(x.shape)
        return self.fn(self.norm(x), **kwargs)

class FeedForward1(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class FeedForward2(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConv(nn.Module):  # layers.append(block(oup, oup, image_size))
    def __init__(self, inp, oup, image_size, depth, channel_out, downsample=False, expansion=4, stochastic_depth=0.2):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(channel_out)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv1 = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
            )
            self.conv2 = nn.Sequential(
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.conv1 = PreNorm(inp, self.conv1, nn.BatchNorm2d)
        self.conv2 = PreNorm(hidden_dim, self.conv2, nn.BatchNorm2d)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        if self.downsample:
            y = self.conv1(x)
            # print(y.shape)
            z = self.conv2(y)
            return self.proj(self.pool(x)) + self.drop_path(z), self.avgpool(y).view(y.shape[0], -1)
        else:
            # print(x[1].shape)
            y = self.conv1(x[0])
            # print(y.shape)
            z = self.conv2(y)
            return x[0] + self.drop_path(z), self.avgpool(y).view(y.shape[0], -1)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=12, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.to_q = nn.Linear(inp, inner_dim, bias = False)
        self.to_kv = nn.Linear(inp, inner_dim * 2, bias = False)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print(x.shape)
        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias

        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)
        attn = self.attend(dots)

        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, depth, channel_out, heads=12, dim_head=16, downsample=False, dropout=0., stochastic_depth=0.2):
        super().__init__()
        hidden_dim = int(channel_out)
        # print(stochastic_depth)
        self.ih, self.iw = image_size
        self.downsample = downsample
        self.depth = depth

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff1 = FeedForward1(oup, hidden_dim, dropout)
        self.ff2 = FeedForward2(oup, hidden_dim, dropout)
        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff1 = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff1, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff2 = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(hidden_dim, self.ff2, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):

        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x[0] + self.drop_path(self.attn(x[0]))

        y = self.ff1(x)
        z = self.ff2(y)
        # x = x + z
        return x + self.drop_path(z), self.avgpool(y).view(y.shape[0], -1)


class DNM_Linear(nn.Module):
    def __init__(self, input_size, out_size, M):
        super(DNM_Linear, self).__init__()

        DNM_W = torch.randn([out_size, M, input_size])
        self.params = nn.ParameterDict({'DNM_W': nn.Parameter(DNM_W),
                                        })
        self.input_size = input_size
        self.out_size = out_size
        self.M = M

    def normalize(self, data, rate, dim=None):
        total = torch.sum(data, dim=dim, keepdim=True)
        data = torch.mul(torch.div(data, total), rate)
        mean = torch.mean(data, dim=dim, keepdim=True)
        std = torch.std(data, dim=dim, keepdim=True)
        normalized_data = torch.div(torch.sub(data, mean), std)
        return normalized_data

    def reshape(self, weight):
        return weight.transpose(0, 2).reshape(self.input_size, -1).unsqueeze(1)

    def forward(self, x):
        x = torch.einsum('ijk,bjk->ibjk', x, self.params['DNM_W'])
        x = torch.sigmoid(x)
        x = torch.sum(x, 3)
        x = torch.prod(x, 2)
        x = self.normalize(x, rate=1, dim=1)
        return x

class CADNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, M, num_classes=1000,
                 block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}
        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], channels[4],(ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], channels[4], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], channels[4], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], channels[4], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], channels[4], (ih // 32, iw // 32))
        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Sequential(
            nn.LayerNorm(channels[-1]),
            DNM_Linear(channels[-1], num_classes, M=M)
        )
        # self.fc = nn.Linear(channels[-1], num_classes, bias=False)
        self.apply(init_weights)
        self.channels = channels
        self.M = M
    def forward(self, x):
        z = torch.empty(x.shape[0], self.M, self.channels[-1]).to(x.device)
        x = self.s0(x)
        x, z[:, 0, :] = self.s1(x)
        x, z[:, 1, :] = self.s2(x)
        x, z[:, 2, :] = self.s3(x)
        x, _ = self.s4(x)
        z[:, 3, :] = self.pool(x).view(-1, x.shape[1])
        x = self.fc(z)
        return x

    def _make_layer(self, block, inp, oup, depth, channel_out, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, depth = i + 1, channel_out = channel_out, downsample=True))
            else:
                layers.append(block(oup, oup, image_size, depth = i + 1, channel_out = channel_out))
        return nn.Sequential(*layers)

def cadnet_0(M, k, num_classes, image_size):
    num_blocks = [2, 2, 3, 5, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CADNet((image_size, image_size), 3, num_blocks, channels, M=4, num_classes=num_classes)

def cadnet_1(M, k, num_classes, image_size):
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CADNet((image_size, image_size), 3, num_blocks, channels, M=4, num_classes=num_classes)

def cadnet_2(M, k, num_classes, image_size):
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [128, 128, 256, 512, 1024]  # D
    return CADNet((image_size, image_size), 3, num_blocks, channels, M=4, num_classes=num_classes)

def cadnet_3(M, k, num_classes, image_size):
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [192, 192, 384, 768, 1536]  # D
    return CADNet((image_size, image_size), 3, num_blocks, channels, M=4, num_classes=num_classes)

def cadnet_4(M, k, num_classes, image_size):
    num_blocks = [2, 2, 12, 28, 2]  # L
    channels = [192, 192, 384, 768, 1536]  # D
    return CADNet((image_size, image_size), 3, num_blocks, channels, M=4, num_classes=num_classes)

def cadnet_7(M, k, num_classes, image_size):
    num_blocks = [2, 2, 4, 42, 2]  # L
    channels = [192, 256, 512, 1024, 3072]  # D
    return CADNet((image_size, image_size), 3, num_blocks, channels, M=4, num_classes=num_classes)
