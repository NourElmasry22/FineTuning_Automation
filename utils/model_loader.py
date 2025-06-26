from transformers import AutoTokenizer, AutoModelForCausalLM

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

    "prollama": "GreatCaptainNemo/ProLLaMA",  
}

def load_model_and_tokenizer(model_key: str):
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_key}. Supported: {list(MODEL_MAP.keys())}")

    model_name = MODEL_MAP[model_key]
    print(f"[INFO] Loading model: {model_key} from {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer
