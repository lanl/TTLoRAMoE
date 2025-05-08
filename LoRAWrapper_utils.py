import tensorly as tl
import torch, sys
import torch.nn as nn
from functools import partial
import math
tl.set_backend('pytorch')

class LoRALinearWrapper(nn.Module):
    def __init__(self, base_layer, r, alpha, device):
        super().__init__()
        self.base_layer = base_layer
        out_dim, in_dim = base_layer.weight.shape

        self.lora_A = nn.Linear(in_dim, r, bias=False).to(device)
        self.lora_B = nn.Linear(r, out_dim, bias=False).to(device)

        self.alpha = alpha
        self.scaling = alpha / r

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base_layer(x) + self.scaling * self.lora_B(self.lora_A(x))

def wrap_model_with_lora(model, config):
    lora_rank = config["lora_rank"]
    lora_alpha = config["lora_alpha"]
    apply_to_query = True
    apply_to_value = True

    assign_lora = partial(LoRALinearWrapper,
                          r=lora_rank,
                          alpha=lora_alpha,
                          device=config["device"])


    if "llama" in config["model_name"]:
        for layer in model.model.layers:
            if apply_to_query:
                layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
            if apply_to_value:
                layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
    else:
        raise ValueError("Model name not recognized. Please use 'roberta' or 'llama' in the model name.")
    return model
