"""MiniMind 单卡知识蒸馏入口。"""

import os
import runpy


if __name__ == "__main__":
    if "RANK" in os.environ:
        raise RuntimeError("单卡入口不能使用 torchrun，请直接使用 python -m 启动")
    runpy.run_module("aislant_minimind.simple_train.train_distillation", run_name="__main__")
