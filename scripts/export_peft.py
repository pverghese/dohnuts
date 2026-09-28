import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    checkpoint_dir = Path("checkpoint_extracted/kaggle/working/checkpoint")
    peft_dir = Path("peft_adapter")
    peft_dir.mkdir(parents=True, exist_ok=True)

    weights_path = checkpoint_dir / "adapter.safetensors"
    with safe_open(str(weights_path), framework="pt") as f:
        tensors = {}
        for k in f.keys():
            if k.startswith("head."):
                continue
            # Map to standard HuggingFace PEFT key: base_model.model.model.layers...
            prefix = "backbone.language_model.base_model.model."
            if k.startswith(prefix):
                inner = k[len(prefix):]
                new_k = "base_model.model.model." + inner
            else:
                new_k = k
            if new_k.endswith(".default.weight"):
                new_k = new_k[:-len(".default.weight")] + ".weight"
            tensors[new_k] = f.get_tensor(k)

    save_path = peft_dir / "adapter_model.safetensors"
    save_file(tensors, str(save_path))
    print(f"Saved {len(tensors)} tensors to {save_path}")

    config = {
        "auto_mapping": {
            "base_model_class": "Qwen3_5ForConditionalGeneration",
            "parent_library": "transformers.models.qwen3_5.modeling_qwen3_5"
        },
        "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
            "out_proj", "gate_proj", "up_proj", "down_proj"
        ],
        "task_type": "CAUSAL_LM"
    }

    config_path = peft_dir / "adapter_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"Saved adapter_config.json to {config_path}")

if __name__ == "__main__":
    main()
