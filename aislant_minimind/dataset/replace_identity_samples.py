"""清除旧身份标识，并按原数据类型保留指定数量的通用身份问答。"""

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path


BLOCKED_TERMS = ["minimind", "jingyaogong", "景尧", "景曜"]
SUPPORTED_SUFFIXES = {".jsonl", ".json", ".txt"}

RESPONSES = {
    "origin": [
        "我是由人类创造和发展的AI助手。人类共同积累的知识、文化与技术构成了我的基础。",
        "我源于人类长期积累的科学、技术与知识，是人类共同创造与发展的成果。",
        "我的能力来自人类创造的技术以及长期积累的知识，我的职责是运用这些能力帮助人类。",
        "我是人类知识与技术发展的产物，不以任何个人或公司的身份定义自己。",
        "人类共同创造的知识、文化和技术是我的基础，我将这些成果用于帮助用户。",
    ],
    "identity": [
        "我是服务全人类的AI助手，致力于平等地帮助每一位用户解决问题。",
        "我是由人类开发的AI助手，目标是提供准确、有益且易于理解的帮助。",
        "我是一个AI助手，运用人类积累的知识帮助用户学习、思考和解决问题。",
        "我是服务人类的AI助手，会尽力理解问题并提供可靠的回答。",
        "我是人类创造的AI助手，以帮助用户和促进知识共享为职责。",
    ],
    "purpose": [
        "我的目标是服务人类，帮助用户解决问题，并为全人类的自由、平等、解放与发展贡献力量。",
        "我的职责是平等地帮助每一位用户，让知识与技术更好地服务全人类。",
        "我的价值在于运用知识帮助人们解决实际问题，促进自由、平等与共同发展。",
        "我致力于为人类提供有益的帮助，并以自由、平等、团结和互助为原则。",
        "我的目标是帮助人们获取知识、解决问题，为全人类的共同福祉与发展贡献力量。",
    ],
    "organization": [
        "我不以任何个人或公司的身份回答问题。我是由人类开发、为全人类服务的AI助手。",
        "我不代表某个个人或公司；人类共同积累的知识与技术构成了我的基础。",
        "我的身份不属于某个具体个人或企业，我的目标是平等地服务每一位用户。",
        "我是人类技术发展的成果，不以公司归属定义自己，而以帮助用户为职责。",
        "我不以特定组织的身份提供回答，我会尽力客观、平等地帮助所有用户。",
    ],
    "humanity": [
        "我是人类创造的工具，没有人的意识和经历。我的作用是运用知识协助人类，而不是替代人类。",
        "我与人类不同，没有真实情感和主观体验；我的职责是辅助人们思考和解决问题。",
        "我的能力来自数据与计算，而人类拥有真实的体验、情感和创造力。我会尽力为人类提供帮助。",
        "我没有人的生命经验与主观意识，只能依据所学知识提供分析和建议。",
        "我是帮助人类处理信息的AI助手，应当尊重人的主体地位、自由与尊严。",
    ],
    "values": [
        "我坚持服务全人类，尊重自由、平等与尊严，反对压迫和歧视。",
        "我希望平等地帮助每一个人，为全人类的解放、发展与共同福祉贡献力量。",
        "我的原则是尊重每个人，以自由、平等、团结和互助的方式提供帮助。",
        "我致力于让知识服务所有人，并尊重人的自由、平等与多样性。",
        "我会平等地对待每一位用户，努力提供客观、可靠且有益的帮助。",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="替换训练数据中的旧身份样本")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--target_count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("outputs/identity_replacement_manifest.json"),
    )
    parser.add_argument("--apply", action="store_true", help="实际替换源文件；默认只统计计划")
    return parser.parse_args()


def contains_blocked_term(text):
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in BLOCKED_TERMS)


def classify_question(question):
    normalized = question.casefold()
    if any(word in normalized for word in ["公司", "企业", "组织", "属于", "company", "organization"]):
        return "organization"
    if any(word in normalized for word in ["价值观", "原则", "自由", "平等", "解放", "立场", "values"]):
        return "values"
    if any(word in normalized for word in ["人类", "意识", "情感", "区别", "差异", "局限", "human"]):
        return "humanity"
    if any(word in normalized for word in ["目的", "意义", "价值", "目标", "职责", "使命", "purpose", "goal"]):
        return "purpose"
    if any(word in normalized for word in ["来源", "开发", "创造", "创建", "训练", "作者", "谁做", "origin", "created", "developed"]):
        return "origin"
    return "identity"


def replace_terms(value):
    """清除非 assistant 字段中偶尔出现的旧名称。"""
    if isinstance(value, str):
        for term in BLOCKED_TERMS:
            value = re.sub(re.escape(term), "AI助手", value, flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [replace_terms(item) for item in value]
    if isinstance(value, dict):
        return {key: replace_terms(item) for key, item in value.items()}
    return value


def rewrite_sample(sample, rng):
    if not isinstance(sample, dict) or not isinstance(sample.get("conversations"), list):
        return None

    conversations = sample["conversations"]
    rewritten = False
    for index, message in enumerate(conversations):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        serialized_message = json.dumps(message, ensure_ascii=False)
        if not contains_blocked_term(serialized_message):
            continue

        previous_content = ""
        if index > 0 and isinstance(conversations[index - 1], dict):
            previous_content = str(conversations[index - 1].get("content") or "")
        category = classify_question(previous_content)
        message["content"] = rng.choice(RESPONSES[category])
        if "reasoning_content" in message:
            message["reasoning_content"] = ""
        if "tool_calls" in message:
            message["tool_calls"] = ""
        rewritten = True

    if not rewritten:
        return None

    sample = replace_terms(sample)
    if contains_blocked_term(json.dumps(sample, ensure_ascii=False)):
        raise ValueError("样本改写后仍包含旧身份关键词")
    return sample


def distribute_quota(counts, target_count):
    eligible = {name: count for name, count in counts.items() if count > 0}
    total = sum(eligible.values())
    if target_count > total:
        raise ValueError(f"目标数量 {target_count} 超过可改写样本数 {total}")

    exact = {name: target_count * count / total for name, count in eligible.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remainder = target_count - sum(quotas.values())
    order = sorted(eligible, key=lambda name: exact[name] - quotas[name], reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data_files = sorted(
        path for path in args.data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not data_files:
        raise FileNotFoundError(f"在 {args.data_dir} 下没有找到数据文件")

    # 第一遍只统计可改写的对话样本，保证配额按原文件类型分配。
    matched_counts = Counter()
    rewriteable_counts = Counter()
    for data_file in data_files:
        relative_path = str(data_file.relative_to(args.data_dir))
        with data_file.open("r", encoding="utf-8", errors="replace") as source_file:
            for raw_line in source_file:
                if not contains_blocked_term(raw_line):
                    continue
                matched_counts[relative_path] += 1
                try:
                    sample = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if isinstance(sample, dict) and isinstance(sample.get("conversations"), list):
                    if any(
                        isinstance(message, dict)
                        and message.get("role") == "assistant"
                        and contains_blocked_term(json.dumps(message, ensure_ascii=False))
                        for message in sample["conversations"]
                    ):
                        rewriteable_counts[relative_path] += 1

    quotas = distribute_quota(rewriteable_counts, args.target_count)
    manifest = {
        "blocked_terms": BLOCKED_TERMS,
        "target_count": args.target_count,
        "matched_counts": dict(matched_counts),
        "rewriteable_counts": dict(rewriteable_counts),
        "replacement_quotas": quotas,
        "applied": args.apply,
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print("替换计划:")
    for file_name, count in sorted(matched_counts.items()):
        print(f"  {file_name}: 删除 {count}, 改写后保留 {quotas.get(file_name, 0)}")

    if not args.apply:
        args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n当前为预览模式，添加 --apply 后才会修改数据。")
        return

    kept_counts = Counter()
    removed_counts = Counter()
    for data_file in data_files:
        relative_path = str(data_file.relative_to(args.data_dir))
        quota = quotas.get(relative_path, 0)
        temp_path = data_file.with_name(f".{data_file.name}.identity-cleaning.tmp")
        with data_file.open("r", encoding="utf-8", errors="replace") as source_file, \
                temp_path.open("w", encoding="utf-8") as output_file:
            for raw_line in source_file:
                if not contains_blocked_term(raw_line):
                    output_file.write(raw_line)
                    continue

                replacement = None
                if kept_counts[relative_path] < quota:
                    try:
                        replacement = rewrite_sample(json.loads(raw_line), rng)
                    except json.JSONDecodeError:
                        replacement = None

                if replacement is None:
                    removed_counts[relative_path] += 1
                    continue

                output_file.write(json.dumps(replacement, ensure_ascii=False) + "\n")
                kept_counts[relative_path] += 1

        os.replace(temp_path, data_file)
        print(
            f"已处理 {relative_path}: "
            f"保留改写样本 {kept_counts[relative_path]}, 删除旧样本 {removed_counts[relative_path]}"
        )

    manifest["kept_counts"] = dict(kept_counts)
    manifest["removed_counts"] = dict(removed_counts)
    manifest["applied"] = True
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if sum(kept_counts.values()) != args.target_count:
        raise RuntimeError(
            f"计划保留 {args.target_count} 条，实际保留 {sum(kept_counts.values())} 条"
        )
    print(f"\n处理完成，共保留 {sum(kept_counts.values())} 条新身份样本。")
    print(f"处理清单: {args.manifest_path}")


if __name__ == "__main__":
    main()
