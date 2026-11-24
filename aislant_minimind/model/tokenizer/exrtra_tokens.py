"""Aislant MiniMind 需要的一些额外 token"""

# 解码时可以由 skip_special_tokens=True 自动跳过的控制 token
SPECIAL_TOKENS = [
    "<|endoftext|>",           # 文档或完整文本结束标记

    "<|im_start|>",            # 一条对话消息的开始标记，后面通常跟 system/user/assistant 角色
    "<|im_end|>",              # 一条对话消息的结束标记

    "<|object_ref_start|>",    # 被引用物体名称或描述的开始标记
    "<|object_ref_end|>",      # 被引用物体名称或描述的结束标记
    "<|box_start|>",           # 目标检测矩形框坐标的开始标记
    "<|box_end|>",             # 目标检测矩形框坐标的结束标记
    "<|quad_start|>",          # 四边形区域坐标的开始标记，常用于倾斜文本或不规则区域
    "<|quad_end|>",            # 四边形区域坐标的结束标记

    "<|vision_start|>",        # 一段视觉输入的开始标记
    "<|vision_end|>",          # 一段视觉输入的结束标记
    "<|vision_pad|>",          # 通用视觉特征占位符，用于对齐视觉编码后的 token 序列
    "<|image_pad|>",           # 图像特征占位符，实际位置由图像编码器输出替换或展开
    "<|video_pad|>",           # 视频特征占位符，实际位置由视频编码器输出替换或展开

    "<|audio_start|>",         # 一段音频输入的开始标记
    "<|audio_end|>",           # 一段音频输入的结束标记
    "<|audio_pad|>",           # 音频特征占位符，用于对齐音频编码后的 token 序列

    "<tts_pad>",               # 文本转语音任务中的补齐标记
    "<tts_text_bos>",          # TTS 文本序列的开始标记（Beginning Of Sequence）
    "<tts_text_eod>",          # TTS 文本序列或文本数据的结束标记（End Of Data）
    "<tts_text_bos_single>",   # 单句或单段 TTS 文本序列的开始标记
]

# 保持为完整 token，但解码时必须保留的内容标记。
ADDITIONAL_TOKENS = [
    "<tool_call>",             # 模型发起工具调用的开始标记，内部通常保存函数名和参数
    "</tool_call>",            # 模型工具调用的结束标记
    "<tool_response>",         # 工具执行结果的开始标记
    "</tool_response>",        # 工具执行结果的结束标记
    "<think>",                 # 模型思考或推理过程的开始标记
    "</think>",                # 模型思考或推理过程的结束标记
]

# 为未来扩展协议预留位置，避免后续改变模型 embedding 大小。
BUFFER_TOKENS = [f"<|buffer{i}|>" for i in range(1, 10)]

ALL_TOKENS = SPECIAL_TOKENS + ADDITIONAL_TOKENS + BUFFER_TOKENS
