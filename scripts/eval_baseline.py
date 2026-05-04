import json
import argparse
import os
from tqdm import tqdm
from mlx_lm import load, generate

def eval_predictions(predictions, targets):
    total = len(targets)
    parsed = 0
    name_correct = 0
    args_f1_sum = 0
    exact_matches = 0

    for p_str, t_str in zip(predictions, targets):
        try:
            p = json.loads(p_str)
            if isinstance(p, dict):
                parsed += 1
            else:
                p = None  # Valid JSON but not a dict (e.g. a bare string)
        except:
            p = None

        try:
            t = json.loads(t_str)
        except:
            t = {}

        if p and t:
            p_name = p.get('name')
            t_name = t.get('name')
            if p_name == t_name:
                name_correct += 1

            p_args = p.get('arguments', {})
            t_args = t.get('arguments', {})

            # Exact match: name and all arguments must match exactly
            if p_name == t_name and p_args == t_args:
                exact_matches += 1

            # Args F1: computed on (key, value) pairs for semantic accuracy
            p_pairs = set((k, str(v)) for k, v in p_args.items())
            t_pairs = set((k, str(v)) for k, v in t_args.items())

            if len(t_pairs) == 0 and len(p_pairs) == 0:
                args_f1_sum += 1.0
            else:
                intersection = p_pairs & t_pairs
                precision = len(intersection) / len(p_pairs) if len(p_pairs) > 0 else 0
                recall = len(intersection) / len(t_pairs) if len(t_pairs) > 0 else 0
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0
                args_f1_sum += f1

    return {
        "parse_rate": parsed / total,
        "name_acc": name_correct / total,
        "arg_f1": args_f1_sum / total,
        "exact_match": exact_matches / total
    }

def load_few_shot_examples(n=3):
    """Load n examples from training set to use as few-shot demonstrations."""
    if not os.path.exists("data/train.jsonl"):
        return []
    examples = []
    with open("data/train.jsonl") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line)
            msgs = ex["messages"]
            examples.append({
                "user": msgs[1]["content"],
                "assistant": msgs[2]["content"]
            })
    return examples

def build_few_shot_messages(system, user, examples):
    """Build a ChatML message list with few-shot demonstrations."""
    messages = [{"role": "system", "content": system}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": user})
    return messages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prompt", "base", "sft", "dpo"], required=True)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    # Check if data exists
    if not os.path.exists("data/test.jsonl"):
        print("Test data not found. Please run data preparation first.")
        return

    with open("data/test.jsonl", "r") as f:
        test_data = [json.loads(line) for line in f]

    test_data = test_data[:200]

    if args.mode == "prompt":
        model, tokenizer = load(args.model_path)
    elif args.mode == "base":
        model, tokenizer = load(args.model_path)
    elif args.mode in ["sft", "dpo"]:
        model, tokenizer = load(args.model_path, adapter_path=f"adapters_{args.mode}")

    predictions = []
    targets = []

    # For 'prompt' mode use few-shot (2 examples); all other modes use zero-shot
    few_shot_examples = load_few_shot_examples(n=2) if args.mode == "prompt" else []
    mode_label = "few-shot" if args.mode == "prompt" else args.mode
    print(f"Evaluating in {mode_label} mode...")

    for ex in tqdm(test_data):
        msgs = ex["messages"]
        system_content = msgs[0]["content"]
        user_content = msgs[1]["content"]
        target_content = msgs[2]["content"]

        if args.mode == "prompt" and few_shot_examples:
            messages = build_few_shot_messages(system_content, user_content, few_shot_examples)
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = generate(model, tokenizer, prompt=prompt, max_tokens=200)
        predictions.append(out)
        targets.append(target_content)

    metrics = eval_predictions(predictions, targets)

    os.makedirs("report", exist_ok=True)
    if args.mode == "prompt":
        out_file = "report/prompt_baseline.json"
    elif args.mode == "base":
        out_file = "report/baseline_metrics.json"
    else:
        out_file = f"report/{args.mode}_metrics.json"

    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved {out_file}: {metrics}")

    # Save raw predictions for error analysis and qualitative examples
    with open(f"report/{args.mode}_predictions.json", "w") as f:
        json.dump([
            {"target": t, "pred": p, "prompt": m["messages"][1]["content"]}
            for t, p, m in zip(targets, predictions, test_data)
        ], f, indent=4)

if __name__ == "__main__":
    main()
