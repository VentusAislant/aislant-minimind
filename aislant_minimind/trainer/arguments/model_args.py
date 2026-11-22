from dataclasses import dataclass


@dataclass
class ModelArguments:
    """所有模型通用的加载参数。"""

    model_path: str | None = None
    tokenizer_path: str | None = None
    trust_remote_code: bool = False


@dataclass
class MiniMindLlamaArguments(ModelArguments):
    """从头创建 MiniMindLlama 时使用的模型结构参数。"""

    hidden_size: int = 768
    num_hidden_layers: int = 8

    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int | None = None

    max_position_embeddings: int = 32768
    rope_theta: float = 1e6

    dropout: float = 0.0
    flash_attn: bool = True

    use_moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 1
    moe_intermediate_size: int | None = None
    router_aux_loss_coef: float = 5e-4