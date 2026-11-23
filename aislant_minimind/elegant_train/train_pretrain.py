import sys
from pathlib import Path

# 可以支持 python xxx/train_pretrain.py 直接运行
__package__ = "aislant_minimind.elegant_train"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, HfArgumentParser

from ..dataset.lm_dataset import PretrainDataset
from ..model.modeling_minimind_llama import (
    MiniMindLlamaConfig,
    MiniMindLlamaForCausalLM,
)
from .arguments import (
    DataArguments,
    MiniMindLlamaArguments,
    PretrainArguments,
    ModelArguments
)
from .base_trainer import Trainer
from .utils import get_model_params


class PretrainTrainer(Trainer):
    stage = "Pretrain"

    def setup_model(self):
        args: ModelArguments = self.model_args

        # 加载已有的 Hugging Face 模型。
        if args.model_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
                trust_remote_code=args.trust_remote_code,
            )
            self.model = MiniMindLlamaForCausalLM.from_pretrained(
                args.model_path,
            )

        # 从 tokenizer 和模型参数开始预训练。
        else:
            if args.tokenizer_path is None:
                raise ValueError("从头预训练必须指定 tokenizer_path")

            self.tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path,
                trust_remote_code=args.trust_remote_code,
            )

            config_kwargs = {
                "vocab_size": len(self.tokenizer),
                "pad_token_id": self.tokenizer.pad_token_id,
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "hidden_size": args.hidden_size,
                "num_hidden_layers": args.num_hidden_layers,
                "num_attention_heads": args.num_attention_heads,
                "num_key_value_heads": args.num_key_value_heads,
                "max_position_embeddings": args.max_position_embeddings,
                "rope_theta": args.rope_theta,
                "dropout": args.dropout,
                "flash_attn": args.flash_attn,
                "use_moe": args.use_moe,
                "num_experts": args.num_experts,
                "num_experts_per_tok": args.num_experts_per_tok,
                "router_aux_loss_coef": args.router_aux_loss_coef,
            }

            if args.intermediate_size is not None:
                config_kwargs["intermediate_size"] = args.intermediate_size

            if args.moe_intermediate_size is not None:
                config_kwargs["moe_intermediate_size"] = args.moe_intermediate_size

            self.model_config = MiniMindLlamaConfig(**config_kwargs)
            self.model = MiniMindLlamaForCausalLM(self.model_config)

        self.model_config = self.model.config
        self.model = self.model.to(self.device)
        get_model_params(self.model, self.model_config)

    def build_dataset(self):
        if self.data_args.data_path is None:
            raise ValueError("必须指定 data_path")

        return PretrainDataset(
            data_path=self.data_args.data_path,
            tokenizer=self.tokenizer,
            max_length=self.data_args.max_seq_len,
        )


if __name__ == "__main__":
    parser = HfArgumentParser(
        (
            MiniMindLlamaArguments,
            DataArguments,
            PretrainArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    trainer = PretrainTrainer(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    trainer.train()
