.PHONY: setup data prompt-baseline baseline sft sft-eval analyze dpo-data dpo dpo-eval report demo

setup:
	uv venv
	uv pip install -r pyproject.toml

data:
	uv run python scripts/prepare_data.py

prompt-baseline:
	uv run python scripts/eval_baseline.py --mode prompt

baseline:
	uv run python scripts/eval_baseline.py --mode base

sft:
	uv run mlx_lm.lora --config config_sft.yaml

sft-eval:
	uv run python scripts/eval_baseline.py --mode sft

analyze:
	uv run python scripts/analyze_errors.py

dpo-data:
	uv run python scripts/prepare_dpo_data.py

dpo:
	uv run python scripts/train_dpo.py

dpo-eval:
	uv run python scripts/eval_baseline.py --mode dpo

report:
	uv run python scripts/generate_report.py

demo:
	uv run uvicorn api.main:app --reload
