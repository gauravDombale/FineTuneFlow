import json
import os

def analyze(mode):
    pred_file = f"report/{mode}_predictions.json"
    if not os.path.exists(pred_file):
        print(f"No predictions found for {mode}")
        return None
        
    with open(pred_file, "r") as f:
        data = json.load(f)
        
    total = len(data)
    invalid_json = 0
    wrong_name = 0
    missing_args = 0
    extra_args = 0
    
    bad_examples = []
    
    for item in data:
        t_str = item["target"]
        p_str = item["pred"]
        prompt = item["prompt"]
        
        try:
            t = json.loads(t_str)
        except:
            t = {}
            
        try:
            p = json.loads(p_str)
        except Exception as e:
            invalid_json += 1
            if len(bad_examples) < 10:
                bad_examples.append({"error": "invalid_json", "pred": p_str, "target": t_str, "prompt": prompt})
            continue
            
        if p.get("name") != t.get("name"):
            wrong_name += 1
            if len(bad_examples) < 10:
                bad_examples.append({"error": "wrong_name", "pred": p_str, "target": t_str, "prompt": prompt})
            continue
            
        t_args = set(t.get("arguments", {}).keys())
        p_args = set(p.get("arguments", {}).keys())
        
        missing = t_args - p_args
        extra = p_args - t_args
        
        if missing:
            missing_args += 1
            if len(bad_examples) < 10:
                bad_examples.append({"error": "missing_args", "missing": list(missing), "pred": p_str, "target": t_str, "prompt": prompt})
        
        if extra:
            extra_args += 1
            if len(bad_examples) < 10:
                bad_examples.append({"error": "extra_args", "extra": list(extra), "pred": p_str, "target": t_str, "prompt": prompt})

    res = {
        "invalid_json_pct": invalid_json / total,
        "wrong_name_pct": wrong_name / total,
        "missing_args_pct": missing_args / total,
        "extra_args_pct": extra_args / total,
        "bad_examples": bad_examples
    }
    return res

def main():
    modes = ["base", "sft"]
    for m in modes:
        res = analyze(m)
        if res:
            with open(f"report/error_analysis_{m}.json", "w") as f:
                json.dump(res, f, indent=4)
            print(f"Saved error analysis for {m}")

if __name__ == "__main__":
    main()
