import argparse
import torch


def build_parser(description="MiniMind Training", **defaults):
    """
    创建带有所有通用训练参数的 parser。

    defaults: 覆盖某些参数的默认值，例如 build_parser(save_weight="pretrain")。
    如果某个阶段还有专属参数，拿到 parser 后自己继续 add_argument 即可。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--save_dir", type=str, default="../outputs", help="模型保存目录")
    parser.add_argument("--tokenizer_path", type=str, default="model_weights/tokenizers/llama",
                        help="Tokenizer 路径")
    parser.add_argument("--save_weight", type=str, default="model", help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--hidden_size", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default=None, help="训练数据路径")
    parser.add_argument("--from_weight", type=str, default="none", help="基于哪个权重训练，为none则从头开始")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind", help="wandb项目名")
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1],
                        help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"], help="学习率调度方式")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="预热步数占总步数的比例，0 表示不预热")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="最低学习率占初始学习率的比例")
    parser.set_defaults(**defaults)
    return parser


def parse_args(
        description="MiniMind Training",
        argv=None,
        **defaults
):
    """解析命令行参数并返回 args。各训练阶段直接调用，只需传入自己不同的默认值。"""
    return build_parser(description, **defaults).parse_args(argv)
