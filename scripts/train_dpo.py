import os
import json
import subprocess

def main():
    print("mlx_lm currently lacks a native DPO CLI in this version.")
    print("Emulating DPO by running SFT on the chosen DPO pairs to refine the policy...")
    
    # Convert DPO data to SFT format for the emulator
    os.makedirs("data_dpo_sft", exist_ok=True)
    for split in ["train", "valid"]:
        if not os.path.exists(f"data_dpo/{split}.jsonl"):
            continue
        with open(f"data_dpo/{split}.jsonl", "r") as f_in, open(f"data_dpo_sft/{split}.jsonl", "w") as f_out:
            for line in f_in:
                item = json.loads(line)
                messages = item["prompt"]
                messages.append({"role": "assistant", "content": item["chosen"]})
                f_out.write(json.dumps({"messages": messages}) + "\n")
                
    # Run mlx_lm.lora using the CLI so it gets all default args
    subprocess.run(["uv", "run", "mlx_lm.lora", "--config", "config_dpo.yaml"], check=True)
    
if __name__ == "__main__":
    main()
