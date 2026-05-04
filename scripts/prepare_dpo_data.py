import json
import os
import random

def create_rejected(target_json_str):
    try:
        data = json.loads(target_json_str)
    except:
        return "{}"
        
    # Corrupt the data
    corruption_type = random.choice(["wrong_name", "missing_arg", "invalid_json", "extra_arg"])
    
    if corruption_type == "wrong_name":
        data["name"] = data.get("name", "") + "_wrong"
        return json.dumps(data)
        
    elif corruption_type == "missing_arg":
        args = data.get("arguments", {})
        if args:
            keys = list(args.keys())
            del args[keys[0]]
            data["arguments"] = args
        else:
            data["name"] = data.get("name", "") + "_wrong"
        return json.dumps(data)
        
    elif corruption_type == "extra_arg":
        args = data.get("arguments", {})
        args["extra_made_up_arg"] = "value"
        data["arguments"] = args
        return json.dumps(data)
        
    elif corruption_type == "invalid_json":
        # Remove a quote or bracket
        s = json.dumps(data)
        return s[:-2] if len(s) > 2 else "{"
        
    return "{}"

def main():
    if not os.path.exists("data/train.jsonl"):
        print("Run data prep first!")
        return
        
    with open("data/train.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
        
    dpo_data = []
    # We need 300 train and 50 valid
    for ex in data[:350]:
        messages = ex["messages"]
        system_msg = messages[0]
        user_msg = messages[1]
        assistant_msg = messages[2]
        
        target_str = assistant_msg["content"]
        rejected_str = create_rejected(target_str)
        
        # mlx_lm DPO format:
        # prompt: list of messages, chosen: string, rejected: string
        # OR: prompt as string. But mlx_lm handles chat template for list of messages.
        
        dpo_data.append({
            "prompt": [system_msg, user_msg],
            "chosen": target_str,
            "rejected": rejected_str
        })
        
    os.makedirs("data_dpo", exist_ok=True)
    
    train_dpo = dpo_data[:300]
    valid_dpo = dpo_data[300:350]
    
    with open("data_dpo/train.jsonl", "w") as f:
        for item in train_dpo:
            f.write(json.dumps(item) + "\n")
            
    with open("data_dpo/valid.jsonl", "w") as f:
        for item in valid_dpo:
            f.write(json.dumps(item) + "\n")
            
    print("Saved data_dpo/train.jsonl (300) and data_dpo/valid.jsonl (50)")

if __name__ == "__main__":
    main()
