import torch
from torch import nn, Tensor
from torch.nn import functional as f

# class MLP(nn.Module):
#     def __init__(self, dim, embed_dim):
#         super().__init__()
#         self.proj = nn.Linear(dim, embed_dim)
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = x.flatten(2).transpose(1, 2)
#         x = self.proj(x)
#         return x


MIT_SETTINGS = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2], [4, 8, 16, 32]],  # [embed_dims, depths, img_size reduction]
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2], [4, 8, 16, 32]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3], [4, 8, 16, 32]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3], [4, 8, 16, 32]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3], [4, 8, 16, 32]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3], [4, 8, 16, 32]]
}


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i + 1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features):
        b, _, h, w = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(b, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i + 2}")(feature).permute(0, 2, 1).reshape(b, -1, *feature.shape[-2:])
            outs.append(f.interpolate(cf, size=(h, w), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, h, w) -> Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.head, c // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(b, c, h, w)
            x = self.sr(x).reshape(b, c, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = self.kv(x).reshape(b, -1, 2, self.head, c // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, h, w) -> Tensor:
        b, _, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: Tensor, h, w) -> Tensor:
        return self.fc2(f.gelu(self.dwconv(self.fc1(x), h, w)))


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, patch_size // 2)  # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor, h, w) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class MiT(nn.Module):
    def __init__(self, model_name: str = 'B0'):
        super().__init__()
        assert model_name in MIT_SETTINGS.keys(), f"MiT model name should be in {list(MIT_SETTINGS.keys())}"
        embed_dims, depths, self.img_size_reductions = MIT_SETTINGS[model_name]
        self.embed_dims = embed_dims
        self.depths = depths
        drop_path_rate = 0.1
        self.channels = embed_dims

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur + i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur + i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur + i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur + i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        b = x.shape[0]
        # stage 1
        x, h, w = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h, w)
        x1 = self.norm1(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)

        # stage 2
        x, h, w = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, h, w)
        x2 = self.norm2(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)

        # stage 3
        x, h, w = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, h, w)
        x3 = self.norm3(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)

        # stage 4
        x, h, w = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, h, w)
        x4 = self.norm4(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)

        return x1, x2, x3, x4


class UnetDecoder(nn.Module):
    def __init__(self, embed_dims, img_size_reductions, n_classes):
        super().__init__()
        embed_dims = sorted(embed_dims, reverse=True)
        embed_dims_pad = [0] + embed_dims
        self.blocks = nn.ModuleList([
            VGGBlock(embed_dims_pad[i] + embed_dims_pad[i + 1], embed_dims_pad[i + 1], embed_dims_pad[i + 1])
            for i in range(len(embed_dims_pad) - 1)
        ])
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.last_layer = nn.Conv2d(embed_dims[-1], n_classes, kernel_size=1)
        self.img_size_reductions = img_size_reductions
        if isinstance(img_size_reductions, list):
            self.img_size_reductions = min(img_size_reductions)

    def forward(self, xs):
        x = xs[0]
        for i in range(len(self.blocks)):
            if i == 0:
                x = self.blocks[i](xs[i])
            else:
                x = self.blocks[i](torch.cat([self.up_sample(x), xs[i]], dim=1))
        x = self.last_layer(x)
        out = f.interpolate(x, scale_factor=self.img_size_reductions, mode='bilinear', align_corners=True)
        return out


class BaseModel(nn.Module):
    def __init__(self, backbone: str = 'MiT-B0'):
        super().__init__()
        backbone, variant = backbone.split('-')
        self.backbone = eval(backbone)(variant)

    def init_pretrained(self, pretrained):
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)


if __name__ == "__main__":
    pass

