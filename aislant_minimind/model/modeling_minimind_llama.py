import math
import torch
import torch.nn.functional as F

from torch import nn
from transformers.activations import ACT2FN
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


###############################
###  1. Config
###############################

class MiniMindLlamaConfig(PretrainedConfig):
    model_type = "aislant-minimind-llama"

    def __init__(
            self,
            hidden_size=768,
            num_hidden_layers=8,
            use_moe=False,
            **kwargs
    ):
        super().__init__(**kwargs)

        # 模型规模参数
        self.use_moe = use_moe
        self.hidden_size = hidden_size  # 模型的隐藏维度(宽度), 也是每个 token 向量的维度
        self.num_hidden_layers = num_hidden_layers  # transformer 层数
        self.vocab_size = kwargs.get("vocab_size", 32001)  # Llama 32000 词表 + 1 个 pad token
        self.intermediate_size = kwargs.get("intermediate_size",
                                            math.ceil(hidden_size * math.pi / 64) * 64)  # 前馈网络的中间层宽度

        # 注意力结构参数
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)  # 注意力头数量
        self.num_key_value_heads = kwargs.get("num_key_value_heads",
                                              4)  # k v 头数, 不指定则默认是 attention head，这里默认设置了为一半 (4/8) 的 attention head 数量,  这样 kv 的头数就比 q 少，使用 kv cache 时缓存会减半
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 每个头的纬度
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否使用 PyTorch 的融合注意力算子，只影响速度和显存，不改变计算结果

        # 位置编码参数
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        # 归一化、正则和激活
        self.dropout = kwargs.get("dropout", 0.0)
        self.hidden_act = kwargs.get("hidden_act", 'silu')  # 前馈网络的激活函数（和 gate、up 组成 SwiGLU）
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)  # RMSNorm 里防止除以 0 的小常数
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # 输入嵌入和输出层共用权重

        ### MoE 特定参数 (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)  # 专家个数
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)  # 每个 token 激活几个专家
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # 每个专家的宽度
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)  # 是否对选中专家的权重做归一化
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 负载均衡损失的系数：太小专家会"偏科"（有的总被选中），太大会拖累主任务

        # 特殊 token
        self.pad_token_id = kwargs.get("pad_token_id", 32000)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)


###############################
###  2. Model
###############################

# 2.1 归一化
class RMSNorm(torch.nn.Module):
    """
    按均方根缩放的归一化
        y = weight * x / sqrt( mean(x²) + eps )
    其中
        x: 正在处理的数据, 也就是一批 token 的向量表示, 形状为 (bsz, seq_len, hidden_size)
        weight: 形状为 (hidden_size) 的可学习向量
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        # rsqrt 为平方根倒数
        # (bsz, seq_len, hidden_size) * (bsz, seq_len, 1)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 转化成 float32 计算来提高数值稳定性, 最后恢复成输入的数据类型
        return (self.weight * self.norm(x.float())).type_as(x)


# 2.2 Attention
class Attention(nn.Module):
    def __init__(self, config: MiniMindLlamaConfig):
        super().__init__()

        # 注意力头配置
        self.num_attention_heads = config.num_attention_heads  # Query Head 数量
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads 必须大于 0")

        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )  # Key/Value Head 数量，如果没有指定，则与 Query Head 数量相同，即普通 MHA

        self.head_dim = config.head_dim  # 每个注意力头维度

        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads 必须大于 0")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads  # 每组 k/v 服务多少个 query head

        # 自回归模型使用因果注意力
        self.is_causal = True

        # Q K V 线性投影
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # Q K-Norm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)  # 用于输出投影后的结果
        self.dropout = config.dropout  # SDPA 接口需要一个 float 类型的 dropout 概率

        # Flash Attention / SDPA 配置
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(
            self, x,
            position_embeddings,
            past_key_value=None,
            use_cache=False,
            attention_mask=None
    ):
        bsz, seq_len, _ = x.shape
        # 使用默认的 num_attention_head=8, num_key_value_head=4, head_dim=96
        # 生成 QKV  xq: [bsz, seq_len, 8*96], xk / xv: [bsz, seq_len, 4*96]
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 拆分注意力头  xq: [bsz, seq_len, 8, 96], xk / xv: [bsz, seq_len, 4, 96]
        xq = xq.view(bsz, seq_len, self.num_attention_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # QK-Norm 每个 head 都单独归一化
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # RoPE 位置编码
        cos, sin = position_embeddings  # 提前计算好的旋转位置编码
        xq, xk = self.apply_rotary_pos_emb(xq, xk, cos, sin)  # 将位置编码应用到 Q 和 K 上
        if past_key_value is not None:  # 拼接历史 KV Cache
            # past_key_value[0]: [bsz, past_len, 4, 96]
            # 后面记 key_len = past_len + seq_len
            xk = torch.cat([past_key_value[0], xk], dim=1)  # [bsz, key_len, 4, 96]
            xv = torch.cat([past_key_value[1], xv], dim=1)  # [bsz, key_len, 4, 96]
        past_kv = (xk, xv) if use_cache else None  # 准备更新 kv cache

        # 扩展 GQA 的 K/V head，attention 计算之前使得 q k v 的 head 数量相同
        xq, xk, xv = (
            xq.transpose(1, 2),  # [bsz, 8, seq_len, 96]
            # 重复前 K0 K1 K2 K3, 重复后 K0 K0 K1 K1 K2 K2 K3 K3
            self.repeat_kv(xk, self.num_key_value_groups).transpose(1, 2),  # [bsz, 8, key_len, 96]
            self.repeat_kv(xv, self.num_key_value_groups).transpose(1, 2)  # [bsz, 8, key_len, 96]
        )

        # attention 计算
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (
                attention_mask is None or torch.all(attention_mask == 1)):
            # 需要满足如下条件:
            #    1. self.flash: 配置允许使用 flash, 存在 scaled_dot_product_attention 计算
            #    2. seq_len > 1: 不是单个 token 的推理
            #    3. not self.is_causal or past_key_value is None: 没有 KV Cache
            #    4. attention_mask is None or torch.all(attention_mask == 1)): 没有 padding 或者 attention_mask 全部有效
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal
            )  # 使用 PyTorch SDPA
        else:  # 使用 手动计算注意力
            # [bsz, 8, seq_len, key_len]
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)

            # 添加 causal mask
            if self.is_causal:
                """
                1. 创建上三角的负无穷矩阵
                    [0, -∞, -∞, -∞]
                    [0,  0, -∞, -∞]
                    [0,  0,  0, -∞]
                    [0,  0,  0,  0]
                2. 加到原来的 mask 上, 使用 -seq_len 可以兼容 kv cache
                """
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len),
                    float("-inf"),
                    device=scores.device).triu(1)

            # 添加 padding mask
            if attention_mask is not None:
                # attention_mask: [bsz, key_len], 其中是 bool 类型
                # attention_mask.to(scores.dtype).unsqueeze(1).unsqueeze(2): [bsz, 1, 1, key_len]
                # 计算后如果是 1 则 score + 0, 如果是 0 则 score - 1e9
                scores += (1.0 - attention_mask.to(scores.dtype).unsqueeze(1).unsqueeze(2)) * -1e9

            # softmax 并聚合 V
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv

        # 合并注意力头: [bsz, 8, seq_len, 96] -> [bsz, seq_len, 8*96]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)

        # 输出投影
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

    @staticmethod
    def apply_rotary_pos_emb(
            q,  # [bsz, seq_len, n_head, head_dim]
            k,  # [bsz, seq_len, n_kv_head, head_dim]
            cos,  # [seq_len, head_dim]
            sin,  # [seq_len, head_dim]
            unsqueeze_dim=1
    ):
        # 把 RoPE（Rotary Position Embedding）应用到 Query 和 Key 上
        def rotate_half(x):
            return torch.cat(
                (-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]),
                dim=-1
            )

        q_embed = (
                (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
        ).to(q.dtype)

        k_embed = (
                (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
        ).to(k.dtype)
        return q_embed, k_embed

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # 用于 GQA (Grouped Query Attention), query head 多于 key value head, 每个 query 查多个 kv head
        bs, seq_len, num_key_value_heads, head_dim = x.shape
        if n_rep == 1:
            # 不用重复
            return x
        else:
            # 这里假设 num_key_value_head = 4, num_attention_head = 8, n_rep = 2
            # K0, K1, K2, K3 -> K0, K0, K1, K1, K2, K2, K3, K3
            results = x[:, :, :, None, :]
            # [bsz, seq_len, num_key_value_heads, n_rep, head_dim]
            results = results.expand(bs, seq_len, num_key_value_heads, n_rep, head_dim)
            # [bsz, seq_len, num_key_value_heads * n_rep, head_dim]
            return results.reshape(bs, seq_len, num_key_value_heads * n_rep, head_dim)


# 2.3.1 前馈神经网络 FeedForward Network
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindLlamaConfig, intermediate_size: int | None = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 1. x [bsz, seq_len, hidden_size]
        # 2. act_fn & gate_proj(x) [bsz, seq_len, intermediate_size] 门控
        # 3. up_proj(x) [bsz, seq_len, intermediate_size]
        # 4. act_fn & gate_proj(x) * up_proj(x)  [bsz, seq_len, intermediate_size]
        # 5. down_proj [bsz, seq_len, hidden_size]
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# 2.3.2 混合专家模型前馈神经网络 Mixture of Experts FeedForward Network
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindLlamaConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [batch_size × seq_len, hidden_dim]
        scores = F.softmax(self.gate(x_flat), dim=-1)  # [batch_size × seq_len, num_experts]

        # 选择 topk 专家
        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )  # [batch_size × seq_len, num_experts_per_tok]

        # 重新归一化 topk 权重
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1:
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                # k=1: 1.0 forward on purpose, gradient via straight-through
                top1 = torch.topk(
                    F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False
                )[0]
                topk_weight = top1 - top1.detach() + 1.0

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):  # 依次处理每个专家
            mask = (topk_idx == i)  # 找到专家负责的 token 位置
            if mask.any():  # 如果有 token 选择这个专家
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                # weight * expert(token) 找到选择这个专家的 token 进行计算
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:  # 没有 token 选择这个专家
                # DDP 分布式训练中，如果某个专家参数没有参与当前 forward，可能被认为是“未使用参数”，导致梯度同步出现问题
                # 所以这里人为创建一个值为 0 的计算依赖
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # 训练阶段且辅助损失系数大于0, 主要用于控制专家负载均衡，避免部分专家过载，部分专家学不到东西
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)  # 恢复原始形状后返回


# 2.4 Layer
class MiniMindLlamaLayer(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindLlamaConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


# 2.5 Model
class MiniMindLlamaModel(nn.Module):
    def __init__(self, config: MiniMindLlamaConfig):
        super().__init__()
        self.config = config

        # 嵌入层
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # num_hidden_layers 层
        self.layers = nn.ModuleList([
            MiniMindLlamaLayer(l, config) for l in range(self.num_hidden_layers)
        ])

        # 归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 旋转位置编码
        freqs_cos, freqs_sin = self.precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos,
                             persistent=False)  # 将位置编码注册 buffer, buffer 不参与梯度更新，但是会随模型移动到 CPU/GPU
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)  # persistent 表示不保存到模型权重文件

    def forward(
            self,
            input_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            **kwargs
    ):
        bsz, seq_len = input_ids.shape

        if hasattr(past_key_values, 'layers'):
            past_key_values = None

        if not past_key_values:
            past_key_values = [None] * len(self.layers)

        if past_key_values[0] is None:
            start_pos = 0
        else:  # 如果存在历史 KV Cache，通过第 0 层 Key Cache 的序列长度，得到已经处理过的 token 数量
            start_pos = past_key_values[0][0].shape[1]

        # 嵌入 [bsz, seq_len] -> [bsz, seq_len, hidden_size]
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = self.precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling
            )
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_len],
                               self.freqs_sin[start_pos:start_pos + seq_len])

        presents = []  # 存每一层的 kv cache
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终状态归一化
        hidden_states = self.norm(hidden_states)

        # 汇总 MoE 辅助损失
        aux_loss = sum(
            [
                l.mlp.aux_loss
                for l in self.layers
                if isinstance(l.mlp, MOEFeedForward)
            ],
            hidden_states.new_zeros(1).squeeze()  # 创建一个和 hidden_states 位于相同设备、具有相同数据类型的标量 0，作为求和起始值
        )
        return hidden_states, presents, aux_loss

    @staticmethod
    def precompute_freqs_cis(
            dim: int,
            end: int = int(32 * 1024),
            rope_base: float = 1e6,
            rope_scaling: dict = None
    ):
        freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
        if rope_scaling is not None:  # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            orig_max, factor, beta_fast, beta_slow, attn_factor = (
                rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
                rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0),
                rope_scaling.get("attention_factor", 1.0)
            )
            if end / orig_max > 1.0:
                inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
                low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
                ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                                   0,
                                   1)
                freqs = freqs * (1 - ramp + ramp / factor)
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
        return freqs_cos, freqs_sin


class MiniMindLlamaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindLlamaConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindLlamaConfig | None = None):
        self.config = config or MiniMindLlamaConfig()
        super().__init__(self.config)
        self.model = MiniMindLlamaModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(
            self,
            input_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            logits_to_keep=0,
            labels=None,
            **kwargs
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
            **kwargs
        )

        # 通过切片来选择需要计算 logits 的 token 位置, 不必每次都生成所有位置的 logits 可以减少现存占用
        # logits_to_keep = 0: 计算所有位置
        # logits_to_keep = 1: 计算最后1个位置
        # logits_to_keep = N: 计算最后N个位置
        if isinstance(logits_to_keep, int):
            # 当 logits_to_keep=0 时，-0 等于 0，
            # slice(0, None) 表示选择所有位置。
            slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )

    @torch.inference_mode()
    def generate(
            self,
            inputs=None,
            attention_mask=None,
            max_new_tokens=8192,
            temperature=0.85,
            top_p=0.85,
            top_k=50,
            eos_token_id=2,
            streamer=None,
            use_cache=True,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.0,
            **kwargs
    ):
        if do_sample and not 0 < temperature < math.inf: raise ValueError(
            "Sampling temperature must be finite and positive")

        # 获取 input_ids 如果没有直接使用 inputs
        # 一次生成多条回复, 默认是 1
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            # 流式输出
            streamer.put(input_ids.cpu())

        # 生成循环
        for _ in range(max_new_tokens):
            # 已有 kv cache 长度
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # 只输入新的 token
            outputs = self.forward(
                input_ids[:, past_len:],
                attention_mask,
                past_key_values,
                use_cache=use_cache,
                **kwargs
            )
            # 扩展 attention mask, 每生成一个 token 在末尾补一个 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1
            ) if attention_mask is not None else None

            # 取最后一个位置的 logits 采样下一个 token
            if do_sample:
                # temperature > 1 概率分布更加平坦，生成更随机
                logits = outputs.logits[:, -1] / temperature
            else:
                logits = outputs.logits[:, -1]

            # 重复性惩罚, 避免输出很多重复的异常 token
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )

            # top-k 过滤
            if do_sample and top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # top-p 过滤 会保留累计概率达到 top_p 的最小 token 集合
            if do_sample and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                logits[mask.scatter(1, sorted_indices, F.pad(mask[..., :-1], (1, 0)))] = -float('inf')

            if do_sample:
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                )
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full(
                        (next_token.shape[0], 1),
                        eos_token_id
                    ),
                    next_token
                )

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                # 强制将后续 token 设置为 EOS
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break

        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids


AutoConfig.register(
    MiniMindLlamaConfig.model_type,
    MiniMindLlamaConfig,
    exist_ok=True,
)

AutoModelForCausalLM.register(
    MiniMindLlamaConfig,
    MiniMindLlamaForCausalLM,
    exist_ok=True,
)
