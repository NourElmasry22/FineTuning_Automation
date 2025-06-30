from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_MAP = {
    "protgpt2": "nferruz/ProtGPT2",

    "progen2-small": "hugohrban/progen2-small",
    "progen2-medium": "hugohrban/progen2-medium",
    "progen2-large": "hugohrban/progen2-large",
    "progen2-xlarge": "hugohrban/progen2-xlarge",

    "RITA_s": "lightonai/RITA_s",
    "RITA_m": "lightonai/RITA_m",
    "RITA_l": "lightonai/RITA_l", 
    "RITA_xl": "lightonai/RITA_xl",

}

def load_model_and_tokenizer(model_key: str):
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_key}. Supported: {list(MODEL_MAP.keys())}")

    model_name = MODEL_MAP[model_key]
    print(f"[INFO] Loading model: {model_key} from {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        #torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        torch_dtype=torch.float32

    )

    return model, tokenizer
