import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

#Gated Spatial Attention Unit (GSAU)
class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut

# multi-scale large kernel attention (MLKA)
class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError("n_feats must be divisible by 3 for MLKA.")

        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],
                      dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x

#multi-scale attention blocks (MAB)
class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LKA = MLKA(n_feats)
        self.LFE = GSAU(n_feats)

    def forward(self, x):
        x = self.LKA(x)
        x = self.LFE(x)
        return x
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out

class ChannelAttention(nn.Module):  # 通道注意力机制
    def __init__(self, in_planes, scaling=16):  # scaling为缩放比例，
        # 用来控制两个全连接层中间神经网络神经元的个数，一般设置为16，具体可以根据需要微调
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):  # 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class CBAM_Attention(nn.Module):
    def __init__(self, channel, scaling=16, kernel_size=7):
        super(CBAM_Attention, self).__init__()
        self.channelattention = ChannelAttention(channel, scaling=scaling)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


from timm.models.layers import SqueezeExcite

import torch


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self

"""
class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv
"""
class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim,
                           kernel_size=sr_ratio + 3,
                           stride=sr_ratio,
                           padding=(sr_ratio + 3) // 2,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=None, ), )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)
class DynamicConv2d(nn.Module):  ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim,
                       dim // reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1), )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):

        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(B, C, H, W)
class DynamicConv2d(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=3,
                 num_groups=2):
        super(DynamicConv2d, self).__init__()

        # 输入通道数
        self.dim = dim

        # 定义卷积层
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=num_groups
        )

        # 如果需要，您可以添加激活函数和归一化层
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        # 通过卷积层
        x = self.conv(x)
        # 激活函数
        x = self.activation(x)
        # 归一化
        x = self.norm(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, bias, norm_cfg, act_cfg):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.norm = None
        if norm_cfg:
            norm_type = norm_cfg.get('type', 'BN2d')
            if norm_type == 'BN2d':
                self.norm = nn.BatchNorm2d(out_channels)
        self.act = None
        if act_cfg:
            act_type = act_cfg.get('type', 'ReLU')
            if act_type == 'GELU':
                self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class HybridTokenMixer(nn.Module):  # D-Mixer
    def __init__(self,
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Attention(
            dim=dim // 2, num_heads=num_heads, sr_ratio=sr_ratio)

        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.InstanceNorm2d(dim),  # 替换为 InstanceNorm
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(inner_dim),  # 替换为 InstanceNorm
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.InstanceNorm2d(dim),  # 替换为 InstanceNorm
        )

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2, relative_pos_enc)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x  # STE
        return x



"""   
class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        # 新的 proj 模块，使用 ed 和 ed 作为输入和输出通道数
        self.proj = nn.Sequential(
            nn.Conv2d(ed, ed, kernel_size=3, padding=1, groups=ed),  # 深度卷积
            nn.GELU(),  # 激活函数
            nn.BatchNorm2d(ed),  # 批归一化
            nn.Conv2d(ed, ed, kernel_size=1),  # 1x1卷积
            nn.GELU(),
            nn.BatchNorm2d(ed),
            nn.Conv2d(ed, ed, kernel_size=1),  # 再次 1x1卷积，保持通道数不变
            nn.BatchNorm2d(ed),
        )
        self.dim = ed  # 保持 dim 为 ed

    def forward(self, x):
        # 使用新的 proj 模块，并保留 residual 路径
        return self.proj(x) + x  # 残差连接
"""

class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert (hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                #MAB(inp) if use_se else nn.Identity(),  # 替换为 MAB
                #TripletAttention(gate_channels=inp, no_spatial=False) if use_se else nn.Identity(),  # 传入 gate_channels 参数
                #CBAM_Attention(inp, scaling=16) if use_se else nn.Identity(),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
        else:
            assert (self.identity)
            self.token_mixer = nn.Sequential(
                HybridTokenMixer(inp),
                #MAB(inp) if use_se else nn.Identity(),  # 替换为 MAB
                #TripletAttention(gate_channels=inp, no_spatial=False) if use_se else nn.Identity(),  # 传入 gate_channels 参数
                #CBAM_Attention(inp, scaling=16) if use_se else nn.Identity(),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                # pw
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU() if use_hs else nn.GELU(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


from timm.models.vision_transformer import trunc_normal_


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier


class RepViT(nn.Module):
    def __init__(self, cfgs, num_classes=3, distillation=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                                          Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self.classifier = Classfier(output_channel, num_classes, distillation)

    def forward(self, x):
        # x = self.features(x)
        for f in self.features:
            x = f(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)

        x = self.classifier(x)
        return x


from timm.models import register_model


@register_model
def repvit_m0_6(pretrained=False, num_classes = 3, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        [3,   2,  40, 1, 0, 1],## 第一个RepVITBlock，包含 SE 模块
        [3,   2,  40, 0, 0, 1],#第二个RepVITBlock，不包含 SE 模块，Stage 1 结束
        [3,   2,  80, 0, 0, 2],# RepViT block (卷积核大小3x3, 通道数80, 下采样stride=2)
        [3,   2,  80, 1, 0, 1],#Stage 2 是网络中第二个块，第一个 RepVITBlock，包含 SE 模块
        [3,   2,  80, 0, 0, 1],## 第二个 RepVITBlock
        [3,   2,  160, 0, 1, 2],# 使用了3x3卷积，通道数160，下采样，与之前的 Downsample 类似，第二个 Downsample 负责进一步减小特征图的尺寸，并增加通道数。
        [3,   2, 160, 1, 1, 1],#Stage 3 是模型的第三个块，RepVITBlock, 使用 SE 模块
        [3,   2, 160, 0, 1, 1],# RepVITBlock
        [3,   2, 160, 1, 1, 1],# RepVITBlock，使用 SE 模块
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 320, 0, 1, 2],#通道数增加到320，进行下采样， 第三个 Downsample
        [3,   2, 320, 1, 1, 1],## RepVITBlock，最后一个 Stage，负责在高维度上进行深层次的特征提取。
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)
#[卷积核大小, 步幅, 输出通道数, 是否使用SE模块, 是否使用HS激活函数, 重复次数]
