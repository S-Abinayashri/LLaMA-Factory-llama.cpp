"""
Merge the latest Hai Indexer LoRA adapter (trained_with_new_data)
into the base Mistral-7B-Instruct-v0.2 model and save a full HF model.
"""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    base_model = "mistralai/Mistral-7B-Instruct-v0.2"
    lora_dir = "saves/Mistral-7B-Instruct-v0.2/lora/trained_with_new_data"
    # Overwrite / refresh the canonical merged model directory
    out_dir = "exports/hai_indexer_mistral_7b_new_data"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[merge] Base model: {base_model}")
    print(f"[merge] LoRA adapter: {lora_dir}")
    print(f"[merge] Output dir: {out_dir}")

    print("[merge] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("[merge] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_dir)

    print("[merge] Merging LoRA into base weights...")
    model = model.merge_and_unload()  # returns plain transformers model

    print("[merge] Saving merged model (safetensors)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    print("[merge] Done. Merged model saved to:", out_dir)


if __name__ == "__main__":
    main()

