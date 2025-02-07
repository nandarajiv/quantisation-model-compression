import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.random.manual_seed(42)


def get_mem_perp_and_time(model_name, model, data, tokenizer, is_8_bit=False):
    print("Computing metrics for", model_name)
    if not is_8_bit:
        model = model.to(device)

    mem = model.get_memory_footprint() / 1024**3

    encodings = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    max_length = model.config.max_position_embeddings
    stride = 512
    sequence_length = encodings.input_ids.size(1)

    times = []
    nlls = []

    previous_end_location = 0
    for start_location in tqdm(range(0, sequence_length, stride)):
        end_location = min(start_location + max_length, sequence_length)
        target_length = end_location - previous_end_location
        if is_8_bit:
            input_ids = encodings.input_ids[:, start_location:end_location]
        else:
            input_ids = encodings.input_ids[:, start_location:end_location].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-target_length] = -100

        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_ids, labels=target_ids)
            end_time = time.time()

            nll = outputs.loss
            times.append(end_time - start_time)
            nlls.append(nll)

        previous_end_location = end_location
        if end_location == sequence_length:
            break

    average_time = np.mean(times)
    perplexity = torch.exp(torch.stack(nlls).mean())

    return mem, perplexity.item(), average_time


model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)

print("Loading original model.")
original_model = AutoModelForCausalLM.from_pretrained(model_name)

print("Loading 4-bit model.")
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading 8-bit model.")
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)

print("Loading NF4 4-bit model.")
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)
model_nf4 = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=nf4_config
)


models = {
    "Original": original_model,
    "4-bit": model_4bit,
    "8-bit": model_8bit,
    "NF4 4-bit": model_nf4,
}


metrics = []
print("Loading dataset.")
text = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# print("Computing metrics.")
for model_name in models.keys():
    if model_name == "8-bit":
        mem, perp, inf_time = get_mem_perp_and_time(
            model_name, models[model_name], text, tokenizer, is_8_bit=True
        )
    else:
        mem, perp, inf_time = get_mem_perp_and_time(
            model_name, models[model_name], text, tokenizer
        )
    metrics.append([model_name, mem, perp, inf_time])

df = pd.DataFrame(
    metrics, columns=["Model", "Memory (GB)", "Perplexity", "Inference Time (s)"]
)
print(df)
df.to_csv("bnb_model_comparison.csv", index=False)

# save model weights as pt
# for model_name in models.keys():
#     models[model_name].save_pretrained(f"{model_name}_weights")
