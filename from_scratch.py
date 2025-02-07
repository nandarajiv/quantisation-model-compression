import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from tqdm import tqdm
import time
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.random.manual_seed(42)


def get_mem_perp_and_time(model, data, tokenizer):
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


class W8A16LinearLayer(nn.Module):
    def __init__(self, input_features, output_features, bias=True, dtype=torch.float32):
        super().__init__()

        self.register_buffer(
            "int8_weights",
            torch.randint(
                -128, 127, (output_features, input_features), dtype=torch.int8
            ),
        )
        self.register_buffer("scales", torch.randn((output_features), dtype=dtype))

        if bias:
            self.register_buffer("bias", torch.randn((1, output_features), dtype=dtype))
        else:
            self.bias = None

    def forward(self, inputs):
        converted_weights = self.int8_weights.to(inputs.dtype)
        output = F.linear(inputs, converted_weights) * self.scales

        if self.bias is not None:
            output = output + self.bias

        return output

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights / scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales


def replace_linear_layer_with_W8A16Linear_layer_and_quantization(
    module, target, exclude_list
):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in exclude_list]):
            old_bias = child.bias
            old_weights = child.weight

            new_module = target(
                child.in_features,
                child.out_features,
                old_bias is not None,
                child.weight.dtype,
            )

            setattr(module, name, new_module)
            getattr(module, name).quantize(old_weights)

            if old_bias is not None:
                getattr(module, name).bias = old_bias

        else:
            replace_linear_layer_with_W8A16Linear_layer_and_quantization(
                child, target, exclude_list
            )


def replace_few_linear_layer_with_W8A16Linear_layer_and_quantization(
    module, target, exclude_list, max_layers=None, current_count=0
):
    if max_layers is not None and current_count >= max_layers:
        return current_count

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any([x == name for x in exclude_list]):
            if max_layers is None or current_count < max_layers:
                old_bias = child.bias
                old_weights = child.weight

                new_module = target(
                    child.in_features,
                    child.out_features,
                    old_bias is not None,
                    child.weight.dtype,
                )

                setattr(module, name, new_module)
                getattr(module, name).quantize(old_weights)

                if old_bias is not None:
                    getattr(module, name).bias = old_bias

                current_count += 1
                if max_layers is not None and current_count >= max_layers:
                    return current_count
        else:
            current_count = (
                replace_few_linear_layer_with_W8A16Linear_layer_and_quantization(
                    child, target, exclude_list, max_layers, current_count
                )
            )
            if max_layers is not None and current_count >= max_layers:
                return current_count

    return current_count


model_name = "facebook/opt-350m"

print("Loading original model.")
og_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model Architecture:")
print(og_model)

print("Loading full quantization model.")
full_quant_model = AutoModelForCausalLM.from_pretrained(model_name)
replace_linear_layer_with_W8A16Linear_layer_and_quantization(
    full_quant_model, W8A16LinearLayer, ["lm_head"]
)

print("Loading partial quantization model.")
partial_quant_model = AutoModelForCausalLM.from_pretrained(model_name)
replace_few_linear_layer_with_W8A16Linear_layer_and_quantization(
    partial_quant_model, W8A16LinearLayer, ["lm_head"], max_layers=5
)

models = {
    "Original": og_model,
    "Full Quantization": full_quant_model,
    "Partial Quantization": partial_quant_model,
}

metrics = []
print("Loading dataset.")
text = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# print("Computing metrics.")
for model_name in models.keys():
    print(f"Computing metrics for {model_name}.")
    mem, perplexity, inf_time = get_mem_perp_and_time(
        models[model_name], text, tokenizer
    )
    metrics.append((model_name, mem, perplexity, inf_time))

df = pd.DataFrame(
    metrics, columns=["Model", "Memory (GB)", "Perplexity", "Inference Time (s)"]
)
print(df)
df.to_csv("from_scratch_model_comparison.csv", index=False)

# save the model weights
# for model_name in models.keys():
#     models[model_name].save_pretrained(f"{model_name}_model")
