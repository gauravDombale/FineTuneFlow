from datasets import load_dataset
ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
for ex in ds:
    if "functioncall" in ex["chat"]:
        print(repr(ex["chat"]))
        break
