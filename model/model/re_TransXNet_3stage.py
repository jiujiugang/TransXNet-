import os
import math
import copy
import torch
from typing import Type
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from timm.models.registry import register_model
from mmcv.runner.checkpoint import load_checkpoint
from timm.models.layers import DropPath, to_2tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath


try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False
from typing import Optional

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        # Convolution Layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        # Normalization Layer
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # Activation Layer
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # Combine all layers
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        if norm_cfg['type'] == 'GN':
            num_groups = norm_cfg.get('num_groups', 32)  # 默认分成 32 组，您可以根据需要调整
            return nn.GroupNorm(num_groups, num_features, eps=norm_cfg.get('eps', 1e-5))
        elif norm_cfg['type'] == 'BN2d':  # 新增对 BN2d 的支持
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        # Add more normalization types if needed
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")
    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        if act_cfg['type'] == 'GELU':  # 新增对 GELU 激活函数的支持
            return nn.GELU()
        # Add more activation types if needed
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

class CAA(nn.Module):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


#上面添加
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

"""
class FusionConvAttention(nn.Module):#调换空间和通道
    def __init__(self, in_channels, out_channels, reduction=4, factor=4.0):
        super(FusionConvAttention, self).__init__()
        dim = int(out_channels // factor)
        dim = in_channels
        # 1x1卷积
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        # 不同大小卷积核
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)

        # 通道和空间注意力模块
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim, reduction=reduction)

        # 最终1x1卷积，用于调整输出通道
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 空间注意力
        x_fused_s = x * self.spatial_attention(x)

        # 多尺度卷积
        x_1x1 = self.down(x)
        x_3x3 = self.conv_3x3(x_1x1)
        x_5x5 = self.conv_5x5(x_1x1)
        x_7x7 = self.conv_7x7(x_1x1)

        # 合并空间卷积输出
        x_fused = x_3x3 + x_5x5 + x_7x7

        # 通道注意力
        x_fused_c = x_fused * self.channel_attention(x_fused)

        # 最终输出
        x_out = self.up(x_fused_c + x_fused_s)
        return x_out
"""
"""
class FusionConvAttention(nn.Module):#多尺度+通道注意力+空间注意力
    def __init__(self, in_channels, out_channels, reduction=4, factor=4.0):
        super(FusionConvAttention, self).__init__()
        dim = int(out_channels // factor)
        dim = in_channels
        # 1x1卷积
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        # 不同大小卷积核
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)

        # 通道和空间注意力模块
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim, reduction=reduction)

        # 最终1x1卷积，用于调整输出通道
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 多尺度卷积
        x_1x1 = self.down(x)
        x_3x3 = self.conv_3x3(x_1x1)
        x_5x5 = self.conv_5x5(x_1x1)
        x_7x7 = self.conv_7x7(x_1x1)

        # 合并多尺度卷积输出
        x_fused_s = x_3x3 + x_5x5 + x_7x7

        # 通过通道注意力
        x_channel = x_fused_s * self.channel_attention(x_fused_s)

        # 通过空间注意力
        x_spatial = x_channel * self.spatial_attention(x_channel)

        # 最终输出
        x_out = self.up(x_spatial)  # 残差连接或上采样
        return x_out
"""
"""
class FusionConvAttention(nn.Module):#多尺度+空间注意力+通道注意力
    def __init__(self, in_channels, out_channels, reduction=4, factor=4.0):
        super(FusionConvAttention, self).__init__()
        dim = int(out_channels // factor)
        dim = in_channels
        # 1x1卷积
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        # 不同大小卷积核
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)

        # 通道和空间注意力模块
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim, reduction=reduction)

        # 最终1x1卷积，用于调整输出通道
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 多尺度卷积
        x_1x1 = self.down(x)
        x_3x3 = self.conv_3x3(x_1x1)
        x_5x5 = self.conv_5x5(x_1x1)
        x_7x7 = self.conv_7x7(x_1x1)

        # 合并多尺度卷积输出
        x_fused_s = x_3x3 + x_5x5 + x_7x7

        # 通过空间注意力
        x_spatial = x_fused_s * self.spatial_attention(x_fused_s)

        # 通过通道注意力
        x_channel = x_spatial * self.channel_attention(x_spatial)

        # 最终输出
        x_out = self.up(x_channel)  # 残差连接或上采样
        return x_out
"""


class FusionConvAttention(nn.Module):#原始最好
    def __init__(self, in_channels, out_channels, reduction=4, factor=4.0):
        super(FusionConvAttention, self).__init__()
        dim = int(out_channels // factor)
        dim = in_channels
        # 1x1卷积
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        # 不同大小卷积核
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)

        # 通道和空间注意力模块
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim, reduction=reduction)

        # 最终1x1卷积，用于调整输出通道
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 通道注意力
        x_fused_c = x * self.channel_attention(x)

        # 多尺度卷积
        x_1x1 = self.down(x)
        x_3x3 = self.conv_3x3(x_1x1)
        x_5x5 = self.conv_5x5(x_1x1)
        x_7x7 = self.conv_7x7(x_1x1)

        # 合并空间卷积输出
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        # 通道和空间注意力输出加和，最终输出
        x_out = self.up(x_fused_s + x_fused_c)   # 残差连接
        return x_out

#上面是添加的块
class PatchEmbed(nn.Module):#是一个补丁嵌入模块，通过卷积实现输入图像特征的嵌入。
    """Patch Embedding module implemented by a layer of convolution.

    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    Args:
        patch_size (int): Patch size of the patch embedding. Defaults to 16.
        stride (int): Stride of the patch embedding. Defaults to 16.
        padding (int): Padding of the patch embedding. Defaults to 0.
        in_chans (int): Input channels. Defaults to 3.
        embed_dim (int): Output dimension of the patch embedding.
            Defaults to 768.
        norm_layer (module): Normalization module. Defaults to None (not use).
    """

    def __init__(self,
                 patch_size=16,#用于卷积的内核大小，决定了每个补丁的大小。
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,#输出嵌入的维度，默认为 768，决定了输出通道数。
                 norm_layer=dict(type='BN2d'),
                 act_cfg=None, ):
        super().__init__()
        self.proj = ConvModule(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            norm_cfg=norm_layer,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        return self.proj(x)#最终输出的张量形状为 [B, embed_dim, H/stride, W/stride]，表示将输入特征映射到嵌入空间


class Attention(nn.Module):  ### OSRA,模块通过自注意力机制提取全局特征
    def __init__(self, dim,#输入特征的通道数
                 num_heads=1,#注意力头的数量
                 qk_scale=None,#查询和键的缩放系数。若未提供，则默认为 1/√head_dim。
                 attn_drop=0,#注意力分数的丢弃率
                 sr_ratio=1, ):#下采样比率，若 sr_ratio>1，会引入额外的空间下采样卷积操作，减少计算量。
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio#
        self.q = nn.Conv2d(dim, dim, kernel_size=1)#用于进一步处理降采样后的特征。
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)#用于进一步处理降采样后的特征。
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
            self.sr = nn.Identity()#决定是否下采样
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)#是 3x3 深度可分离卷积，应用于经过 OSR 下采样的特征图 kv 上，以增强局部特征

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)#使用 1x1 卷积生成 q，调整形状以分配给各个注意力头，然后进行维度转置，便于后续计算
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv#通过下采样（self.sr）和局部卷积 self.local_conv 处理输入，得到局部增强的 kv。
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)#使用 self.kv(kv) 对处理后的 kv 进行 1x1 卷积，并通过 torch.chunk(..., chunks=2, dim=1) 将其分成两部分，分别赋值给 k 和 v。
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)#使用 self.kv 生成 k 和 v，并 reshape 和 transpose 以匹配多头注意力的格式
        attn = (q @ k) * self.scale#使用矩阵乘法计算 q 与 k 的点积，并乘以缩放系数 self.scale。
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc#若提供了相对位置编码 relative_pos_enc，则根据 attn 的形状调整后相加
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)#应用 softmax 和 Dropout，得到归一化的注意力权重。
        x = (attn @ v).transpose(-1, -2)#计算 attn 与 v 的加权和（自注意力输出）。
        return x.reshape(B, C, H, W)#调整维度并 reshape 成与输入一致的形状 B, C, H, W。
#OSR 部分对应代码中的 self.sr 和 self.local_conv，用于对输入进行降采样。Linear 层用于生成查询、键和值。MHSA 部分实现了多头自注意力机制，用于提取全局特征。

class DynamicConv2d(nn.Module):  ### IDConv是动态卷积模块，为每个输入特征生成一组动态卷积权重,
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,#通道缩减比例，用于降维。
                 num_groups=1,#分组数，用于生成多组卷积权重。
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)#用于保存 num_groups 个卷积核权重，每个权重的形状为 [dim, kernel_size, kernel_size]
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))#将输入特征图缩小到与卷积核大小一致的分辨率 ([kernel_size, kernel_size])
        self.proj = nn.Sequential(#投影层 proj：用于生成动态卷积核的系数 scale，包括两部分
            ConvModule(dim,
                       dim // reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),#首先用 1x1 卷积将输入通道数减少到 dim // reduction_ratio，然后通过 GELU 激活
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1), )#再次用 1x1 卷积恢复通道数至 dim * num_groups，用于后续的权重生成。

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None#偏置 bias：如果需要偏置，则初始化一个形状为 [num_groups, dim] 的可学习偏置参数。

        self.reset_parameters()#调用 reset_parameters 方法初始化权重和偏置。

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)#初始化 weight 和 bias：使用截断正态分布对 weight 和 bias 进行初始化，标准差为 0.02，确保初始参数较小且符合正态分布

    def forward(self, x):

        B, C, H, W = x.shape#获取输入特征图的批次、通道、高度和宽度。
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)#池化和投影：通过 self.pool 池化并使用 self.proj 生成卷积核的动态权重比例 scale。重塑 scale：调整 scale 的形状为 [B, num_groups, C, K, K]，其中每个组的通道和卷积核大小与输入一致。
        scale = torch.softmax(scale, dim=1)#归一化 scale：对 scale 进行 softmax 处理，确保每组权重在不同组间归一化。
        weight = scale * self.weight.unsqueeze(0)#将 self.weight 扩展至批次维度（增加维度 [1, num_groups, C, K, K]），以便与 scale 相乘
        weight = torch.sum(weight, dim=1, keepdim=False)#按分组维度 dim=1 汇总生成每个通道的动态卷积权重（输出维度 [B, C, K, K]）。
        weight = weight.reshape(-1, 1, self.K, self.K)#调整形状为 [B * C, 1, K, K] 以用于分组卷积操作

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)#若启用偏置，则计算全局均值池化后的 scale，并 softmax 归一化。
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)#重塑输入 x 的形状为 [1, B * C, H, W] 以进行分组卷积，groups=B * C 允许对每个通道使用独立的卷积核，使用动态生成的 weight 和 bias 执行 F.conv2d，得到卷积结果。

        return x.reshape(B, C, H, W)#输出形状恢复：调整卷积结果的形状，恢复到 [B, C, H, W]。
class HybridTokenMixer(nn.Module):#IDConv2d  +  FusionConvAttention
    def __init__(self,
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 norm_cfg=None,  # 用于传递规范化配置
                 act_cfg=None,#新加
                 reduction_ratio=8,
                 drop_path=0.0):  # 添加 drop_path 参数
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)
        self.caa = CAA(channels=dim, h_kernel_size=11, v_kernel_size=11, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.global_unit = FusionConvAttention(
            in_channels=dim // 2, out_channels=dim // 2, reduction=reduction_ratio)

        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )


    def forward(self, x, relative_pos_enc=None):
        # 将输入特征图 x 分成两部分
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        # 局部特征处理：使用 DynamicConv2d 处理 x1
        x1 = self.local_unit(x1)



        # 全局特征处理：使用 FusionConvAttention 处理 x2
        x2 = self.global_unit(x2)

        # 拼接局部和全局特征
        x = torch.cat([x1, x2], dim=1)
        # 投影与残差连接

        x = self.proj(x) + x
        return x

"""
class HybridTokenMixer(nn.Module):  ### D-Mixer模块结合 DynamicConv2d 和 Attention，进行局部和全局特征的混合#yuanshi
    def __init__(self,
                 dim,#输入特征的通道数。确保它是偶数（因为要分成两部分）
                 kernel_size=3,#局部动态卷积核的大小。
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,#用于注意力的降采样比例。
                 reduction_ratio=8):#用于通道数降维的比例。
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)#局部单元 local_unit：创建 DynamicConv2d 层，使用输入通道的一半（dim // 2）进行动态卷积操作。
        self.global_unit = Attention(
            dim=dim // 2, num_heads=num_heads, sr_ratio=sr_ratio)#全局单元 global_unit：创建 Attention 层，也使用 dim // 2 的通道数，并通过多头注意力机制处理全局特征。

        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=3, padding=2, dilation=2),  # 空洞卷积
            #nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), )#投影单元 proj：由一系列卷积、激活函数、和归一化层构成，用于对经过局部和全局单元处理后的特征进一步变换。深度卷积：3x3 的深度卷积处理输入，保留通道维度（dim）
#降维和扩展：通过 1x1 卷积将通道降至 inner_dim（最小值为 16，保证通道数不至于过小），再用 1x1 卷积恢复到原始通道数 dim
    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)#将输入特征图 x 沿通道维度 dim=1 分成两部分，分别作为局部和全局处理的输入 x1 和 x2。
        x1 = self.local_unit(x1)#局部特征处理：使用 local_unit（即 DynamicConv2d）处理 x1，获得局部特征
        x2 = self.global_unit(x2, relative_pos_enc)#全局特征处理：使用 global_unit（即 Attention）处理 x2，获得全局特征。relative_pos_enc 用于相对位置编码，增强空间感知能力。
        x = torch.cat([x1, x2], dim=1)#将局部和全局特征在通道维度拼接，恢复到与输入相同的通道数。
        x = self.proj(x) + x  ## STE，投影与残差连接：通过 proj 层对融合特征进行变换，再通过残差连接（+ x）保留部分原始信息，增强训练稳定性。
        return x
"""

class MultiScaleDWConv(nn.Module):#模块主要用于多尺度深度卷积，将这些不同的尺度特征融合在一起
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)#通道分配：对 dim 进行分割，使每个尺度的卷积核处理不同的通道数量。
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],#卷积核大小由 scale 指定的值确定。
                             padding=scale[i] // 2,#确保卷积输出的宽高不变
                             groups=channels)#进行深度卷积，每个通道独立卷积。
            self.channels.append(channels)
            self.proj.append(conv)#当前卷积核处理的通道数和卷积层添加到 self.channels 和 self.proj 中，便于后续调用。

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)#通道分割：根据 self.channels 中定义的通道数，将输入特征 x 在通道维度上进行分割，以匹配每个卷积层的输入。
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))#逐尺度卷积：遍历每个分割的特征部分 feat，通过相应的卷积层（self.proj[i]）进行处理，并将结果存储在 out 列表中
        x = torch.cat(out, dim=1)
        return x#特征拼接：将所有尺度卷积处理后的结果在通道维度拼接，形成包含多尺度特征的输出 x


class Mlp(nn.Module):  ### MS-FFN旨在利用多尺度卷积和前馈网络来增强特征表达能力。它在通道维度上引入多尺度特征，
    """
    Mlp implemented by with 1x1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='Sigmoid'),  # 修改默认激活函数为 Sigmoid
                 #act_cfg=dict(type='GELU'),
                 drop=0, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),#1x1 卷积层，用于将输入特征映射到 hidden_features 维度。
            build_activation_layer(act_cfg),#激活函数，使用 act_cfg 中指定的激活函数（默认是 GELU）。
            nn.BatchNorm2d(hidden_features),#批归一化层，有助于提高训练的稳定性。
        )
        self.dwconv = MultiScaleDWConv(hidden_features)#通过 MultiScaleDWConv 实现多尺度特征提取操作，输入为 hidden_features，使得特征具有多尺度的局部信息。
        self.act = build_activation_layer(act_cfg)#设置一个单独的激活函数 self.act。
        self.norm = nn.BatchNorm2d(hidden_features)#self.norm 用于进一步对多尺度卷积的输出进行归一化。
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )#输出层 fc2：另一个 1x1 卷积和批归一化模块，将 hidden_features 的输出恢复到 in_features，保持输入输出维度一致。
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)#第一层卷积：通过 fc1 将输入 x 的通道数变为 hidden_features，并激活和归一化。

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))#多尺度卷积和残差连接

        x = self.drop(x)#通过 dropout 对多尺度卷积后的结果进行失活。
        x = self.fc2(x)#将特征映射回原来的通道数。
        x = self.drop(x)#再次进行 dropout。


        return x


class LayerScale(nn.Module):#模块是一个用于通道权重调整的层，通过对每个通道分别引入可学习的缩放系数（weight）和偏置（bias），从而让模型更灵活地控制每个通道特征的幅度和偏移。
    def __init__(self, dim, init_value=1e-5):#初始化的权重缩放因子的值，默认值为
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value,
                                   requires_grad=True)#创建一个具有维度 [dim, 1, 1, 1] 的权重参数，初始值为 init_value。此参数会对每个通道的输入进行缩放。
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)#创建一个具有维度 [dim] 的偏置参数，初始值为 0。它会按通道添加到输出上。

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])#使用 conv2d 函数对输入 x 进行卷积运算，但这里的卷积核和偏置被设定为 self.weight 和 self.bias。groups=x.shape[1] 表示每个通道独立进行卷积（深度卷积），因此 weight 和 bias 会分别作用于每个通道
        return x#

class Block(nn.Module):#yuanshi
    """
    Network Block.
    Args:
        dim (int): Embedding dim.
        kernel_size (int): kernel size of dynamic conv. Defaults to 3.
        num_groups (int): num_groups of dynamic conv. Defaults to 2.
        num_heads (int): num_groups of self-attention. Defaults to 1.
        mlp_ratio (float): Mlp expansion ratio. Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='GN', num_groups=1)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-5.
    """

    def __init__(self,
                 dim=64,
                 kernel_size=3,
                 sr_ratio=1,
                 num_groups=2,
                 num_heads=1,
                 mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop=0,
                 drop_path=0,
                 layer_scale_init_value=1e-5,

                 grad_checkpoint=False):

        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)#计算 MLP 的隐藏层维度，等于嵌入维度与扩展比例的乘积
        # 直接使用 MBConv
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)#位置编码 pos_embed：使用一个7x7卷积，通过深度卷积对每个通道独立处理，用于提取空间位置信息。
        self.caa = CAA(channels=dim, h_kernel_size=11, v_kernel_size=11, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)  # 普通卷积
        self.DWconv3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # Depthwise Convolution
            nn.Conv2d(dim, dim, kernel_size=1)  # Pointwise Convolution
        )
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]#规范化层 norm1：通过 norm_cfg 配置构建归一化层。
        self.token_mixer = HybridTokenMixer(dim,#token 混合器 token_mixer：使用 HybridTokenMixer 结合卷积和Attention进行特征混合。
                                            kernel_size=kernel_size,
                                            num_groups=num_groups,
                                            num_heads=num_heads,
                                            sr_ratio=sr_ratio)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]#规范化层 norm1：通过 norm_cfg 配置构建归一化层。
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_cfg=act_cfg,
                       drop=drop, )#使用 Mlp 类来进行进一步的特征处理。
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()#随机深度丢弃，用于在训练过程中随机丢弃一些特征通道

        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale(dim, layer_scale_init_value)
            self.layer_scale_2 = LayerScale(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()

    def _forward_impl(self, x, relative_pos_enc=None):
        # 位置编码
        x = x + self.pos_embed(x)
        # 原始前向传播路径：归一化、token_mixer、残差连接
        x = x + self.drop_path(self.layer_scale_1(
            self.token_mixer(self.norm1(x), relative_pos_enc)))
        x = self.caa(x) + x
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))
        return x

    def forward(self, x, relative_pos_enc=None):
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_impl, x, relative_pos_enc)
        else:
            x = self._forward_impl(x, relative_pos_enc)
        return x


"""
    def _forward_impl(self, x, relative_pos_enc=None):
        x = x + self.pos_embed(x)#对输入 x 进行卷积位置编码，并与原始输入 x 相加。
        x = x + self.drop_path(self.layer_scale_1(
            self.token_mixer(self.norm1(x), relative_pos_enc)))#先对输入 x 进行归一化处理，再通过 token_mixer 进行特征混合，接着经过层缩放、随机深度丢弃，并与原始输入相加
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))#类似地，norm2 后通过 MLP，经过 layer_scale_2 缩放、丢弃，最后相加得到最终输出。
        return x

    def forward(self, x, relative_pos_enc=None):
        if self.grad_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_impl, x, relative_pos_enc)
        else:
            x = self._forward_impl(x, relative_pos_enc)#若开启梯度检查点，则使用 checkpoint 节省显存，否则直接调用 _forward_impl。
        return x
"""

def basic_blocks(dim,#函数的作用是根据指定的参数创建一个包含多个 Block 实例的模块列表 (nn.ModuleList)，用于构建模型中的一个阶段。每个 Block 实例代表网络中的一个处理单元，结合卷积、注意力、MLP等机制处理输入特征。
                 index,
                 layers,#当前阶段的索引，用于确定 Block 的数量。
                 kernel_size=3,#卷积核大小，默认值为 3。
                 num_groups=2,#
                 num_heads=1,
                 sr_ratio=1,
                 mlp_ratio=4,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop_rate=0,
                 drop_path_rate=0,
                 layer_scale_init_value=1e-5,
                 grad_checkpoint=False):#是否启用梯度检查点，控制计算图保存以节省内存。
    blocks = nn.ModuleList()#创建一个 ModuleList 容器 blocks 用于存储多个 Block 实例。ModuleList 方便进行多层 Block 的顺序存储和调用。
    for block_idx in range(layers[index]):#循环创建当前阶段的 Block 数量（由 layers[index] 确定
        block_dpr = drop_path_rate * (       #对每个 Block，按一定规则计算其 drop_path 值 (block_dpr)，使用线性增加的策略为不同 Block 分配不同的丢弃概率，使模型具有更好的正则化效果。
                block_idx + sum(layers[:index])) / (sum(layers) - 1)#循环次数为当前阶段的 Block 数量。
        blocks.append(
            Block(
                dim,
                kernel_size=kernel_size,
                num_groups=num_groups,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop=drop_rate,
                drop_path=block_dpr,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=grad_checkpoint,
            ))#将每个新创建的 Block 实例添加到 blocks 模块列表中，每个 Block 的参数在实例化时会传入具体的超参数。
    return blocks


class TransXNet(nn.Module):
    """
    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``arch_settings``. And if dict, it
            should include the following two keys:

            - layers (list[int]): Number of blocks at each stage.
            - embed_dims (list[int]): The number of channels at each stage.
            - mlp_ratios (list[int]): Expansion ratio of MLPs.
            - layer_scale_init_value (float): Init value for Layer Scale.

            Defaults to 'tiny'.

        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        in_patch_size (int): The patch size of input image patch embedding.
            Defaults to 7.
        in_stride (int): The stride of input image patch embedding.
            Defaults to 4.
        in_pad (int): The padding of input image patch embedding.
            Defaults to 2.
        down_patch_size (int): The patch size of downsampling patch embedding.
            Defaults to 3.
        down_stride (int): The stride of downsampling patch embedding.
            Defaults to 2.
        down_pad (int): The padding of downsampling patch embedding.
            Defaults to 1.
        drop_rate (float): Dropout rate. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        grad_checkpoint (bool): Using grad checkpointing for saving memory.
        checkpoint_stage (Sequence | bool): Decide which layer uses grad checkpointing. 
                                            For example, checkpoint_stage=[0,0,1,1] means that stage3 and stage4 use gd
        out_indices (Sequence | int): Output from which network position.
            Index 0-6 respectively corresponds to
            [stage1, downsampling, stage2, downsampling, stage3, downsampling, stage4]
            Defaults to -1, means the last stage.#从网络的哪个位置输出。索引 0-6 分别对应于 [stage1, downsampling, stage2, downsampling, stage3, downsampling, stage4]。默认值为 -1，表示最后一个阶段。
        frozen_stages (int): Stages to be frozen (all param fixed).#要冻结的阶段（所有参数固定）。默认值为 0，表示不冻结任何参数。


            Defaults to 0, which means not freezing any parameters.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501

    # --layers: [x,x,x,x], numbers of layers for the four stages
    # --embed_dims, --mlp_ratios:
    #     embedding dims and mlp ratios for the four stages
    # --downsamples: flags to apply downsampling or not in four blocks
    arch_settings = {
        **dict.fromkeys(['t', 'tiny', 'T'], {
            'layers': [1, 1, 1],  # 修改为3个阶段
            'embed_dims': [48, 96, 224],
            'kernel_size': [7, 7, 7],
            'num_groups': [2, 2, 2],
            'sr_ratio': [8, 4, 2],
            'num_heads': [1, 2, 4],
            'mlp_ratios': [4, 4, 4],
            'layer_scale_init_value': 1e-5,
        }),
        **dict.fromkeys(['s', 'small', 'S'], {
            'layers': [3, 4, 6],
            'embed_dims': [64, 128, 320],
            'kernel_size': [7, 7, 7],
            'num_groups': [2, 2, 3],
            'sr_ratio': [8, 4, 2],
            'num_heads': [1, 2, 5],
            'mlp_ratios': [6, 6, 4],
            'layer_scale_init_value': 1e-5,
        }),
        **dict.fromkeys(['b', 'base', 'B'], {
            'layers': [4, 8, 16],
            'embed_dims': [96, 192, 384],
            'kernel_size': [7, 7, 7],
            'num_groups': [2, 3, 4],
            'sr_ratio': [8, 4, 2],
            'num_heads': [2, 4, 8],
            'mlp_ratios': [8, 8, 4],
            'layer_scale_init_value': 1e-5,
        }),
    }

    def __init__(self,
                 image_size=28,
                 arch='small',
                 norm_cfg=dict(type='GN', num_groups=1),#归一化配置，
                 act_cfg=dict(type='GELU'),#激活函数配置
                 in_chans=3,
                 in_patch_size=7,
                 in_stride=4,#在输入图像上滑动块时的步幅
                 in_pad=3,#输入图像的填充大小，在对图像进行卷积或划分块时，边缘可能需要填充以保持尺寸。
                 down_patch_size=3,#在下采样时的块大小，通常用于特征图的进一步处理。
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0,
                 drop_path_rate=0,
                 grad_checkpoint=False,#如果设置为 True，在前向传播时会存储中间激活值，以节省内存，但在反向传播时需要重新计算这些激活值
                 checkpoint_stage=[0] * 4,#指定在每个阶段应用检查点的层级，通常用于控制模型的内存使用。
                 num_classes=3,
                 fork_feat=False,#如果设置为 True，可能表示在特定层的特征会被分叉，以便在不同的路径中使用
                 start_level=0,#指定从哪个层开始进行处理，通常用于多尺度特征学习。
                 init_cfg=None,#初始化配置，通常用于权重初始化，指定如何加载预训练模型等
                 pretrained=None,#预训练模型的路径或标识符，如果提供，则会加载相应的权重。
                 **kwargs):

        super().__init__()
        '''
        The above image_size does not need to be adjusted, 
        even if the image input size is not 224x224,
        unless you want to change the size of the relative positional encoding
        '''
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.grad_checkpoint = grad_checkpoint
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]#如果 arch 是字符串，使用它在 self.arch_settings 中查找对应的架构配置，并将结果赋值给 arch 变量
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "layers" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        kernel_size = arch['kernel_size']
        num_groups = arch['num_groups']
        sr_ratio = arch['sr_ratio']
        num_heads = arch['num_heads']#这段代码主要是对模型的初始化过程进行设置，确保输入参数的有效性，并提取出模型架构相关的配置参数。这些参数将在模型的前向传播和训练过程中使用。

        if not grad_checkpoint:
            checkpoint_stage = [0] * 3

        mlp_ratios = arch['mlp_ratios'] if 'mlp_ratios' in arch else [4, 4, 4]#从 arch 字典中提取 mlp_ratios，如果不存在，则将其初始化为 [4, 4, 4, 4]。这个变量通常表示多层感知器（MLP）的比率，影响隐藏层的大小
        layer_scale_init_value = arch['layer_scale_init_value'] if 'layer_scale_init_value' in arch else 1e-5#从 arch 字典中提取 layer_scale_init_value，如果不存在，则将其初始化为 1e-5。这个值通常用于初始化某些层的缩放因子

        self.patch_embed = PatchEmbed(patch_size=in_patch_size,
                                      stride=in_stride,
                                      padding=in_pad,
                                      in_chans=in_chans,
                                      embed_dim=embed_dims[0])#创建 PatchEmbed 对象并将其赋值给 self.patch_embed

        self.relative_pos_enc = []#把每个阶段生成的相对位置编码添加到 relative_pos_enc 列表中，方便在模型的不同层调用。

        self.pos_enc_record = []#这两个列表将用于存储相对位置编码和位置编码的记录。
        image_size = to_2tuple(image_size)
        image_size = [math.ceil(image_size[0] / in_stride),#将 image_size 转换为二元组形式，通常用于确保图像尺寸是宽度和高度的形式
                      math.ceil(image_size[1] / in_stride)]#根据 in_stride 计算新的 image_size。这一步将图像的宽度和高度分别除以步幅并向上取整，得到经过嵌入后的图像尺寸。
        for i in range(3):#启动一个循环，循环四次（通常与模型层数有关）。
            num_patches = image_size[0] * image_size[1]#计算当前图像尺寸下的补丁数量，即宽度和高度的乘积
            sr_patches = math.ceil(
                image_size[0] / sr_ratio[i]) * math.ceil(image_size[1] / sr_ratio[i])#根据当前的空间缩放比，计算经过缩放后的补丁数量。
            self.relative_pos_enc.append(
                nn.Parameter(torch.zeros(1, num_heads[i], num_patches, sr_patches), requires_grad=True))#DPE（动态位置编码）--使用 nn.Parameter 创建位置编码参数，用于存储相对位置编码，张量的维度为 (1, num_heads[i], num_patches, sr_patches)，并将其添加到 self.relative_pos_enc 列表中。这个张量的梯度将被计算（requires_grad=True）。
            self.pos_enc_record.append([image_size[0], image_size[1],
                                        math.ceil(image_size[0] / sr_ratio[i]),
                                        math.ceil(image_size[1] / sr_ratio[i]), ])#将当前的 image_size 以及缩放后的尺寸（经过 sr_ratio[i] 缩放）以列表的形式添加到 self.pos_enc_record 中。
            image_size = [math.ceil(image_size[0] / 2),
                          math.ceil(image_size[1] / 2)]#每次迭代结束时，将 image_size 除以 2 并向上取整，以便在下一轮计算中使用，通常用于下采样的情况。
        self.relative_pos_enc = nn.ParameterList(self.relative_pos_enc)#这个对应着DPE---relative_pos_enc 是一个 nn.ParameterList，其中每个元素都是相对位置编码张量 nn.Parameter，用于存储四个阶段的相对位置编码。
        # self.relative_pos_enc = [None] * 4

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(#调用 basic_blocks 函数创建一个网络阶段 stage，传入一系列参数，
                embed_dims[i],
                i,
                layers,
                kernel_size=kernel_size[i],
                num_groups=num_groups[i],
                num_heads=num_heads[i],
                sr_ratio=sr_ratio[i],
                mlp_ratio=mlp_ratios[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                grad_checkpoint=checkpoint_stage[i], )#调用 basic_blocks 函数创建一个网络阶段 stage，传入相应的参数
            network.append(stage)#将创建的 stage 添加到 network 列表中。
            if i >= len(layers) - 1:
                break#如果当前的索引 i 已经到达或超过 layers 列表的最后一个元素，提前结束循环。这通常是为了防止在最后一个阶段后再进行下采样。
            if embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1]))#检查当前阶段的嵌入维度是否与下一个阶段不同。如果不同，表示需要进行下采样。通过创建 PatchEmbed 模块将其添加到网络中，以便在两个阶段之间调整特征图的大小。
        self.network = nn.ModuleList(network)#将构建的网络模块列表转换为 nn.ModuleList
        if self.fork_feat:#检查 fork_feat 属性，用于决定是否需要为每个输出添加规范化层。
            # add a norm layer for each output
            self.out_indices = [0, 2, 4]#定义 self.out_indices，表示要输出的层的索引，通常用于选择中间层的输出。
            for i_emb, i_layer in enumerate(self.out_indices):#遍历 self.out_indices 中的索引和对应的层索引。
                if i_emb < start_level:
                    layer = nn.Identity()#如果当前索引 i_emb 小于 start_level，则将 layer 设置为 nn.Identity()，即不对输入做任何变换。
                else:
                    layer = build_norm_layer(norm_cfg, embed_dims[(i_layer + 1) // 2])[1]#否则，根据 norm_cfg 和嵌入维度构建规范化层，通常是 BatchNorm 或 LayerNorm。
                layer_name = f'norm{i_layer}'#生成层的名称，格式为 norm{i_layer}。
                self.add_module(layer_name, layer)#将构建的规范化层添加到模型中，使其成为模型的一部分
        else:
            # Classifier
            self.classifier = nn.Sequential(
                build_norm_layer(norm_cfg, embed_dims[-1])[1],
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1),
            ) if num_classes > 0 else nn.Identity()#如果 fork_feat 为 False，则构建分类器。分类器包括最后一个嵌入维度的规范化层、自适应平均池化和一个卷积层，输出类别数为 num_classes

        self.apply(self._init_model_weights)#应用 _init_model_weights 方法对模型的权重进行初始化。
        self.init_cfg = copy.deepcopy(init_cfg)#将 init_cfg 进行深拷贝，存储在 self.init_cfg 中。
        # load pre-trained model
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()#检查 fork_feat 是否为 True，并且 init_cfg 或 pretrained 不是 None，如果条件满足，则加载预训练模型
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)
            self.train()#将模型设置为训练模式。这会启用 Dropout 和 BatchNorm 等训练特有的功能。

    # init for image classification
    def _init_model_weights(self, m):#初始化不同类型层的权重和偏置。
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    '''
    init for mmdetection or mmsegmentation 
    by loading imagenet pre-trained weights
    '''


    def get_classifier(self):
        return self.classifier#返回当前模型的分类器部分。

    def reset_classifier(self, num_classes):#法用于在训练或推理过程中根据新的类别数量更新分类器。
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier[-1].out_channels = num_classes
        else:
            self.classifier = nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x#方法主要负责将输入图像转换为嵌入特征。

    def forward_tokens(self, x):#forward_tokens 方法负责将嵌入特征通过多个网络层进行前向传播，并根据设置决定是否返回特定层的特征。
        """
        通过多个网络层对嵌入特征进行前向传播，并根据条件决定返回哪些特征。

        参数:
            x: 输入的嵌入特征张量。

        返回:
            特征列表或最后一层的特征张量。
        """
        outs = []
        pos_idx = 0

        # 遍历网络中的所有层
        for idx, layer in enumerate(self.network):
            if idx in [0, 2, 4]:
                # 处理特定层并应用相对位置编码
                for blk in layer:
                    x = blk(x, self.relative_pos_enc[pos_idx])
                pos_idx += 1
            else:
                # 直接通过网络层
                x = layer(x)

            # 如果需要输出特征且当前层在指定输出索引中
            if self.fork_feat and (idx in self.out_indices):
                x_out = getattr(self, f'norm{idx}')(x)
                outs.append(x_out)

        # 返回特征
        if self.fork_feat:
            return outs  # 返回所有关键层特征
        return x  # 返回最后一层特征用于分类

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)

        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # features of four stages for dense prediction
            return x
        else:
            # for image classification
            x = self.classifier(x).flatten(1)
            return x

