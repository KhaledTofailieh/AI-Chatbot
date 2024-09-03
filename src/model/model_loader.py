import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

def load_hf_model(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if device != "cpu":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id, )

    pipeline_args = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "do_sample": True,
        "return_full_text":False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    pipe = pipeline(
        "text-generation",
        # "question-answering",
        model=model,
        tokenizer=tokenizer,
        **pipeline_args
    )
    return model, tokenizer, pipe
