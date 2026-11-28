# Adapted from Microsoft LoRA. Copyright (c) Microsoft Corporation.

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


class LoRALayer():
    """
    混入类 (Mixin): 主要用来保存 LoRA 的超参数和状态标志
    让后面的 Linear, Embedding, MergedLinear, ConvLoRA 通过继承的方式复用
    """
    def __init__(
            self,
            r: int,   # lora rank
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

    def merge(self):
        """将 LoRA 增量合并到原始权重。"""
        raise NotImplementedError

    def unmerge(self):
        """从原始权重中移除已经合并的 LoRA 增量。"""
        raise NotImplementedError


class Embedding(nn.Embedding, LoRALayer):
    """
    LoRA 化的 Embedding
    nn.Embedding 的权重形状是 [num_embeddings, embedding_dim] 主要用于 token 嵌入
    num_embeddings 就是词表大小, embedding_dim 是词向量维度
    """
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            r: int = 0,
            lora_alpha: int = 1,
            merge_weights: bool = True,
            **kwargs
    ):
        # 初始化两个父类
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)  # 先调用原始的初始化
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            # 这里的 AB 初始化策略和 Linear 相反
            """
            原因是 Embedding 的"输入"是 one-hot,A 相当于一张查表矩阵,这里作者选择让 A 为 0、B 随机,
            效果上都能保证起点增量为 0,梯度也能流动(B 非零时 A 能获得梯度;之后 A 变非零,B 也能更新)
            """
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        # model.train() 调用 train(True),model.eval() 调用 train(False)
        # 因为 train 和 eval 可以重复调用，这里需要合并标记，来防止重复的合并和拆解
        nn.Embedding.train(self, mode)
        if self.merge_weights:
            self.unmerge() if mode else self.merge()
        return self

    @torch.no_grad()
    def merge(self):
        if self.r > 0 and not self.merged:
            # 防止重复合并
            self.weight.add_((self.lora_B @ self.lora_A).transpose(0, 1), alpha=self.scaling)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.r > 0 and self.merged:
            # 防止重复拆解
            self.weight.sub_((self.lora_B @ self.lora_A).transpose(0, 1), alpha=self.scaling)
            self.merged = False

    def forward(self, input: torch.Tensor):
        if self.r > 0 and not self.merged:
            # 启用 lora 且未合并，也就是训练模式
            result = nn.Embedding.forward(self, input)
            after_A = F.embedding(
                input, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )  # [num_embeddings, r]
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            # 没启用 lora 或者 已经合并的评估模式
            return nn.Embedding.forward(self, input)


class Linear(nn.Linear, LoRALayer):
    # LoRA 化的线性层
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        # 默认的 nn.linear 权重是 [out, in], 如果 设置了 fan_in_fan_out 则权重为 [in, out]
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            # 所以初始 $BA = 0$,模型起点与预训练模型完全一致。同时 A 非零,保证第一步反传时 B 就能拿到非零梯度
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights:
            self.unmerge() if mode else self.merge()
        return self

    def _lora_delta_weight(self):
        delta_weight = self.lora_B @ self.lora_A
        return delta_weight.transpose(0, 1) if self.fan_in_fan_out else delta_weight

    @torch.no_grad()
    def merge(self):
        if self.r > 0 and not self.merged:
            self.weight.add_(self._lora_delta_weight(), alpha=self.scaling)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.r > 0 and self.merged:
            self.weight.sub_(self._lora_delta_weight(), alpha=self.scaling)
            self.merged = False

    def forward(self, input: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(input, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(input) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(input, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    """
    在某些模型中(如 GPT2)的注意力 q k v 三个投影层通常被合并成一个更大的 linear
        将 w_q, w_k, w_v 在输出维度上拼起来 -> [3*d, in]
        输出 [..., 3d] 然后切分成 q, k, v
    问题:
        普通 Linear 只能对整个大矩阵加一个 LoRA(B: [3d, r]),等于 q、k、v 共用一个低秩增量
        无法"只给 q 和 v 加 LoRA, LoRA 论文的经典设置恰恰是只对 q 和 v 加
    MergedLinear:
        在一个合并的大矩阵里,选择性地只对其中某几段(q/k/v)加 LoRA,每段各有独立的 A、B
    """
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False],  # 例如: enable_lora=[True, False, True]:权重分 3 段(q, k, v),q 和 v 启用、k 不启用
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        """
        假设 out=3d, r=8,  enable_lora=[True, False, True]
            lora_A: [r*2, in] = [16, in]
            lora_b: [(3d/3)*2, r] = [2d, r]
        """
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            # lora_ind: 标记哪些输出位置属于启用的段
            """
            假设 out=3*2=6, enable_lora=[True, False, True]
                view(3, -1): [[F, F],
                              [F, F],
                              [F, F]]
                置 True 的行 0 和 2:
                             [[T, T],
                              [F, F],
                              [T, T]]
                展平:         [T, T, F, F, T, T]
            """
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        # 对每个启用的段 i, 计算 \Delta_i = B_i A_i, 各段之间互不干扰
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            # 卷积输入: 形状为 [bsz, channel_in, seq_len] 分别对应 batch size, 输入通道数, 序列长度
            input=self.lora_A.unsqueeze(0),  # [1, r*k, in]
            # 卷积参数: 形状为 [channel_out, channel_in, kernel_size]
            weight=self.lora_B.unsqueeze(-1),  # [d*k, r, 1]
            # 分组卷积: 输入通道和输出通道平均分为 groups 组, 第 i 组输出只给第 i 组输入计算
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights:
            self.unmerge() if mode else self.merge()
        return self

    @torch.no_grad()
    def merge(self):
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            self.weight.add_(self.merge_AB(), alpha=self.scaling)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.r > 0 and any(self.enable_lora) and self.merged:
            self.weight.sub_(self.merge_AB(), alpha=self.scaling)
            self.merged = False

    def forward(self, input: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(input, T(self.weight), bias=self.bias)
        else:
            result = F.linear(input, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(input) @ T(self.merge_AB().T) * self.scaling
            return result


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0.,
                 merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        for name, param in self.conv.named_parameters():
            self.register_parameter(name, param)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((out_channels // self.conv.groups * kernel_size, r * kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if self.merge_weights:
            self.unmerge() if mode else self.merge()
        return self

    def _lora_delta_weight(self):
        return (self.lora_B @ self.lora_A).view(self.conv.weight.shape)

    @torch.no_grad()
    def merge(self):
        if self.r > 0 and not self.merged:
            self.conv.weight.add_(self._lora_delta_weight(), alpha=self.scaling)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.r > 0 and self.merged:
            self.conv.weight.sub_(self._lora_delta_weight(), alpha=self.scaling)
            self.merged = False

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)


# Can Extend to other ones like this
class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)
