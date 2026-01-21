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

import torch
import torch.nn as nn
from einops import rearrange


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
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7  # 将三个卷积结果相加，生成一个新的特征图
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out


class FusionConvAttention(nn.Module):
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
        x_out = self.up(x_fused_s + x_fused_c)  # 残差连接
        return x_out


class PatchEmbed(nn.Module):  # 是一个补丁嵌入模块，通过卷积实现输入图像特征的嵌入。
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
                 patch_size=16,  # 用于卷积的内核大小，决定了每个补丁的大小。
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,  # 输出嵌入的维度，默认为 768，决定了输出通道数。
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
        return self.proj(x)  # 最终输出的张量形状为 [B, embed_dim, H/stride, W/stride]，表示将输入特征映射到嵌入空间


class DynamicConv2d(nn.Module):  ### IDConv是动态卷积模块，为每个输入特征生成一组动态卷积权重,
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,  # 通道缩减比例，用于降维。
                 num_groups=1,  # 分组数，用于生成多组卷积权重。
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size),
                                   requires_grad=True)  # 用于保存 num_groups 个卷积核权重，每个权重的形状为 [dim, kernel_size, kernel_size]
        self.pool = nn.AdaptiveAvgPool2d(
            output_size=(kernel_size, kernel_size))  # 将输入特征图缩小到与卷积核大小一致的分辨率 ([kernel_size, kernel_size])
        self.proj = nn.Sequential(  # 投影层 proj：用于生成动态卷积核的系数 scale，包括两部分
            ConvModule(dim,
                       dim // reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),  # 首先用 1x1 卷积将输入通道数减少到 dim // reduction_ratio，然后通过 GELU 激活
            nn.Conv2d(dim // reduction_ratio, dim * num_groups,
                      kernel_size=1), )  # 再次用 1x1 卷积恢复通道数至 dim * num_groups，用于后续的权重生成。

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None  # 偏置 bias：如果需要偏置，则初始化一个形状为 [num_groups, dim] 的可学习偏置参数。

        self.reset_parameters()  # 调用 reset_parameters 方法初始化权重和偏置。

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias,
                                  std=0.02)  # 初始化 weight 和 bias：使用截断正态分布对 weight 和 bias 进行初始化，标准差为 0.02，确保初始参数较小且符合正态分布

    def forward(self, x):

        B, C, H, W = x.shape  # 获取输入特征图的批次、通道、高度和宽度。
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K,
                                                self.K)  # 池化和投影：通过 self.pool 池化并使用 self.proj 生成卷积核的动态权重比例 scale。重塑 scale：调整 scale 的形状为 [B, num_groups, C, K, K]，其中每个组的通道和卷积核大小与输入一致。
        scale = torch.softmax(scale, dim=1)  # 归一化 scale：对 scale 进行 softmax 处理，确保每组权重在不同组间归一化。
        weight = scale * self.weight.unsqueeze(0)  # 将 self.weight 扩展至批次维度（增加维度 [1, num_groups, C, K, K]），以便与 scale 相乘
        weight = torch.sum(weight, dim=1, keepdim=False)  # 按分组维度 dim=1 汇总生成每个通道的动态卷积权重（输出维度 [B, C, K, K]）。
        weight = weight.reshape(-1, 1, self.K, self.K)  # 调整形状为 [B * C, 1, K, K] 以用于分组卷积操作

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)  # 若启用偏置，则计算全局均值池化后的 scale，并 softmax 归一化。
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)  # 重塑输入 x 的形状为 [1, B * C, H, W] 以进行分组卷积，groups=B * C 允许对每个通道使用独立的卷积核，使用动态生成的 weight 和 bias 执行 F.conv2d，得到卷积结果。

        return x.reshape(B, C, H, W)  # 输出形状恢复：调整卷积结果的形状，恢复到 [B, C, H, W]。


class HybridTokenMixer(nn.Module):  # IDConv2d  +  FusionConvAttention
    def __init__(self,
                 dim,
                 kernel_size=3,
                 num_groups=2,
                 num_heads=1,
                 sr_ratio=1,
                 reduction_ratio=8,
                 drop_path=0.0):  # 添加 drop_path 参数
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)
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
        # x = self.rcm(x) + x
        x = self.proj(x) + x
        return x


class MultiScaleDWConv(nn.Module):  # 模块主要用于多尺度深度卷积，将这些不同的尺度特征融合在一起
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)  # 通道分配：对 dim 进行分割，使每个尺度的卷积核处理不同的通道数量。
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],  # 卷积核大小由 scale 指定的值确定。
                             padding=scale[i] // 2,  # 确保卷积输出的宽高不变
                             groups=channels)  # 进行深度卷积，每个通道独立卷积。
            self.channels.append(channels)
            self.proj.append(conv)  # 当前卷积核处理的通道数和卷积层添加到 self.channels 和 self.proj 中，便于后续调用。

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels,
                        dim=1)  # 通道分割：根据 self.channels 中定义的通道数，将输入特征 x 在通道维度上进行分割，以匹配每个卷积层的输入。
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))  # 逐尺度卷积：遍历每个分割的特征部分 feat，通过相应的卷积层（self.proj[i]）进行处理，并将结果存储在 out 列表中
        x = torch.cat(out, dim=1)
        return x  # 特征拼接：将所有尺度卷积处理后的结果在通道维度拼接，形成包含多尺度特征的输出 x


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
                 act_cfg=dict(type='GELU'),
                 drop=0, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            # 1x1 卷积层，用于将输入特征映射到 hidden_features 维度。
            build_activation_layer(act_cfg),  # 激活函数，使用 act_cfg 中指定的激活函数（默认是 GELU）。
            nn.BatchNorm2d(hidden_features),  # 批归一化层，有助于提高训练的稳定性。
        )
        self.dwconv = MultiScaleDWConv(
            hidden_features)  # 通过 MultiScaleDWConv 实现多尺度特征提取操作，输入为 hidden_features，使得特征具有多尺度的局部信息。
        self.act = build_activation_layer(act_cfg)  # 设置一个单独的激活函数 self.act。
        self.norm = nn.BatchNorm2d(hidden_features)  # self.norm 用于进一步对多尺度卷积的输出进行归一化。
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
        )  # 输出层 fc2：另一个 1x1 卷积和批归一化模块，将 hidden_features 的输出恢复到 in_features，保持输入输出维度一致。
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  # 第一层卷积：通过 fc1 将输入 x 的通道数变为 hidden_features，并激活和归一化。

        x = self.dwconv(x) + x
        x = self.norm(self.act(x))  # 多尺度卷积和残差连接

        x = self.drop(x)  # 通过 dropout 对多尺度卷积后的结果进行失活。
        x = self.fc2(x)  # 将特征映射回原来的通道数。
        x = self.drop(x)  # 再次进行 dropout。

        return x


class LayerScale(nn.Module):  # 模块是一个用于通道权重调整的层，通过对每个通道分别引入可学习的缩放系数（weight）和偏置（bias），从而让模型更灵活地控制每个通道特征的幅度和偏移。
    def __init__(self, dim, init_value=1e-5):  # 初始化的权重缩放因子的值，默认值为
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value,
                                   requires_grad=True)  # 创建一个具有维度 [dim, 1, 1, 1] 的权重参数，初始值为 init_value。此参数会对每个通道的输入进行缩放。
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)  # 创建一个具有维度 [dim] 的偏置参数，初始值为 0。它会按通道添加到输出上。

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[
            1])  # 使用 conv2d 函数对输入 x 进行卷积运算，但这里的卷积核和偏置被设定为 self.weight 和 self.bias。groups=x.shape[1] 表示每个通道独立进行卷积（深度卷积），因此 weight 和 bias 会分别作用于每个通道
        return x  #


class Block(nn.Module):  # yuanshi
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
        mlp_hidden_dim = int(dim * mlp_ratio)  # 计算 MLP 的隐藏层维度，等于嵌入维度与扩展比例的乘积
        # 直接使用 MBConv
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3,
                                   groups=dim)  # 位置编码 pos_embed：使用一个7x7卷积，通过深度卷积对每个通道独立处理，用于提取空间位置信息。
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)  # 普通卷积
        self.DWconv3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),  # Depthwise Convolution
            nn.Conv2d(dim, dim, kernel_size=1)  # Pointwise Convolution
        )
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]  # 规范化层 norm1：通过 norm_cfg 配置构建归一化层。
        self.token_mixer = HybridTokenMixer(dim,  # token 混合器 token_mixer：使用 HybridTokenMixer 结合卷积和Attention进行特征混合。
                                            kernel_size=kernel_size,
                                            num_groups=num_groups,
                                            num_heads=num_heads,
                                            sr_ratio=sr_ratio)
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]  # 规范化层 norm1：通过 norm_cfg 配置构建归一化层。
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_cfg=act_cfg,
                       drop=drop, )  # 使用 Mlp 类来进行进一步的特征处理。
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()  # 随机深度丢弃，用于在训练过程中随机丢弃一些特征通道

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
        x = x + self.drop_path(self.layer_scale_2(self.mlp(self.norm2(x))))
        return x


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
    def __init__(self,
                 image_size=28,
                 arch='tiny',
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_chans=3,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=3,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0,
                 drop_path_rate=0,
                 grad_checkpoint=False,
                 checkpoint_stage=[0] * 4,
                 num_classes=3,
                 fork_feat=False,
                 start_level=0,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.grad_checkpoint = grad_checkpoint
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "layers" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        kernel_size = arch['kernel_size']
        num_groups = arch['num_groups']
        sr_ratio = arch['sr_ratio']
        num_heads = arch['num_heads']

        if not grad_checkpoint:
            checkpoint_stage = [0] * 4

        mlp_ratios = arch['mlp_ratios'] if 'mlp_ratios' in arch else [4, 4, 4, 4]
        layer_scale_init_value = arch['layer_scale_init_value'] if 'layer_scale_init_value' in arch else 1e-5

        # 修改这里，只保留一个 stage
        self.patch_embed = PatchEmbed(patch_size=in_patch_size,
                                      stride=in_stride,
                                      padding=in_pad,
                                      in_chans=in_chans,
                                      embed_dim=embed_dims[0])

        self.relative_pos_enc = []
        self.pos_enc_record = []
        image_size = to_2tuple(image_size)
        image_size = [math.ceil(image_size[0] / in_stride), math.ceil(image_size[1] / in_stride)]

        # 只需要为第一个stage计算相对位置编码
        num_patches = image_size[0] * image_size[1]
        sr_patches = math.ceil(image_size[0] / sr_ratio[0]) * math.ceil(image_size[1] / sr_ratio[0])
        self.relative_pos_enc.append(
            nn.Parameter(torch.zeros(1, num_heads[0], num_patches, sr_patches), requires_grad=True))
        self.pos_enc_record.append([image_size[0], image_size[1],
                                    math.ceil(image_size[0] / sr_ratio[0]),
                                    math.ceil(image_size[1] / sr_ratio[0])])

        # 网络的构建部分
        network = []
        # 只保留第一个 stage
        stage = basic_blocks(
            embed_dims[0],
            0,  # stage 0
            layers,
            kernel_size=kernel_size[0],
            num_groups=num_groups[0],
            num_heads=num_heads[0],
            sr_ratio=sr_ratio[0],
            mlp_ratio=mlp_ratios[0],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            grad_checkpoint=checkpoint_stage[0],
        )
        network.append(stage)

        # 只有一个 stage，不进行下采样
        self.network = nn.ModuleList(network)

        if self.fork_feat:
            self.out_indices = [0]  # 只输出第一个stage
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb < start_level:
                    layer = nn.Identity()
                else:
                    layer = build_norm_layer(norm_cfg, embed_dims[(i_layer + 1) // 2])[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.classifier = nn.Sequential(
                build_norm_layer(norm_cfg, embed_dims[-1])[1],
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1),
            ) if num_classes > 0 else nn.Identity()

        self.apply(self._init_model_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def _init_model_weights(self, m):
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

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                self.load_state_dict(state_dict)


