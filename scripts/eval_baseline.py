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
            parsed += 1
        except:
            p = None
            
        try:
            t = json.loads(t_str)
        except:
            # Target should be valid JSON
            t = {}
            
        if p and t:
            p_name = p.get('name')
            t_name = t.get('name')
            if p_name == t_name:
                name_correct += 1
                
            p_args = p.get('arguments', {})
            t_args = t.get('arguments', {})
            
            # Simple exact match
            if p_name == t_name and p_args == t_args:
                exact_matches += 1
                
            # Args F1
            p_keys = set(p_args.keys())
            t_keys = set(t_args.keys())
            
            if len(t_keys) == 0 and len(p_keys) == 0:
                args_f1_sum += 1.0
            else:
                intersection = p_keys.intersection(t_keys)
                precision = len(intersection) / len(p_keys) if len(p_keys) > 0 else 0
                recall = len(intersection) / len(t_keys) if len(t_keys) > 0 else 0
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

def build_few_shot_prompt(system, user, examples):
    # This is a very simple few-shot string building
    # In reality, ChatML with few-shot is best
    # Qwen uses ChatML.
    messages = [{"role": "system", "content": system}]
    for ex in examples:
        messages.extend([
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]}
        ])
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
        
    # We only take the first 200 (should be 200 anyway)
    test_data = test_data[:200]
    
    if args.mode == "prompt":
        # For prompt baseline, we'll do 2 setups: zero-shot and few-shot
        # But we need a model. Let's just use the base model.
        model, tokenizer = load(args.model_path)
    elif args.mode == "base":
        model, tokenizer = load(args.model_path)
    elif args.mode in ["sft", "dpo"]:
        # Load LoRA adapters
        model, tokenizer = load(args.model_path, adapter_path=f"adapters_{args.mode}")
        
    predictions_zero = []
    targets = []
    
    print(f"Evaluating in {args.mode} mode...")
    
    # We use temp=0 for eval as requested
    for ex in tqdm(test_data):
        msgs = ex["messages"]
        system_content = msgs[0]["content"]
        user_content = msgs[1]["content"]
        target_content = msgs[2]["content"]
        
        # Zero-shot
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ], tokenize=False, add_generation_prompt=True)
        
        # Generate
        out = generate(model, tokenizer, prompt=prompt, max_tokens=200)
        predictions_zero.append(out)
        targets.append(target_content)
        
    metrics = eval_predictions(predictions_zero, targets)
    
    os.makedirs("report", exist_ok=True)
    out_file = f"report/{args.mode}_metrics.json"
    if args.mode == "prompt":
        out_file = "report/prompt_baseline.json"
    elif args.mode == "base":
        out_file = "report/baseline_metrics.json"
        
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Saved {out_file}: {metrics}")
    
    # Save raw predictions for error analysis
    with open(f"report/{args.mode}_predictions.json", "w") as f:
        json.dump([{"target": t, "pred": p, "prompt": m[1]["content"]} for t, p, m in zip(targets, predictions_zero, [ex["messages"] for ex in test_data])], f, indent=4)

if __name__ == "__main__":
    main()
