import torch.nn as nn
import tensorly as tl
import torch
import torch.nn.functional as F
from typing import Dict
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
import pytorch_lightning as pl
import torchmetrics
from tensorly.decomposition import tensor_train
from functools import partial
from transformers.modeling_outputs import BaseModelOutput
from utils import print_experts
from transformers import AutoModelForSequenceClassification#for llm router
tl.set_backend('pytorch')

def tensorized_multiplication_experts(self, X, tt_cores, m_factors, n_factors, gates):
    """
    - X: [B, m]
    - tt_cores: list of (len(m_factors)+len(n_factors)) cores,
                each core has shape [E, r_i, factor_dim, r_{i+1}],
                where E is the number of experts.
    - gates: [B, E], gating weights for each sample.
      e.g. from a router or gumbel_softmax.
    - returns: [B, prod(n_factors)]
    """
    B = X.shape[0]
    seq_len = X.shape[1]
    shape_x = [B] + [seq_len]+  m_factors[::-1]
    tt_state = X.view(shape_x).unsqueeze(1)  # => [B, 1, ...]
    tt_state.to(self.device)

    num_m = len(m_factors)
    num_n = len(n_factors)
    total_cores = num_m + num_n

    if len(tt_cores) != total_cores:
        raise ValueError(f'Expected {total_cores} TT-cores, got {len(tt_cores)}')
    
    #print("Input data",X[0][0][:5])
    for i in range(num_m):
        core = tt_cores[i].to(self.device)
        gates_exp = gates.view(B,-1,1, 1, 1).to(self.device)
        core_expanded = core.unsqueeze(0).to(self.device)  # [1, 1, E, r_i, m_i, r_{i+1}]
        masked_core_m = (core_expanded * gates_exp).sum(dim=1).to(self.device)
        #print("\nMasked m core\n",masked_core_m[0][0][0][0][0][:5])
        eq_in = "b r ... m, b r m p -> b p ..."
        tt_state = torch.einsum(eq_in, tt_state, masked_core_m).to(self.device)

    start_n = num_m
    for i in range(num_n):
        core = tt_cores[start_n + i].to(self.device)  # shape [E, r_i, n_i, r_{i+1}]
        gates_exp = gates.view(B, -1, 1, 1, 1).to(self.device)
        core_expanded = core.unsqueeze(0).to(self.device)  # => [1, E, r_i, n_i, r_{i+1}]
        masked_core_n = (core_expanded * gates_exp).sum(dim=1).to(self.device)
        #print("\nMasked n core\n",masked_core_n[0][0][0][0][0][:5])
        eq_out = "b r ..., b r n p -> b p ... n"
        tt_state = torch.einsum(eq_out, tt_state, masked_core_n).to(self.device)

    Z = tt_state.view(B, seq_len, -1).to(self.device)
    if self.experts_trainable and self.training:
        Z = self.dropout(Z)
    return Z

class SeparateLLMRouterLightningModule(pl.LightningModule):
    def __init__(self, router_model, lr):
        super().__init__()
        self.learning_rate = lr
        self.router_model = router_model

    def forward(self, input_ids, attention_mask, labels):
        return self.router_model(input_ids, attention_mask=attention_mask, labels=labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class MultiStepRouter(nn.Module):
    def __init__(self, input_dim, num_experts, router_type, num_heads=4, hidden_dim=1024):
        super(MultiStepRouter, self).__init__()
        self.m = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.router_type = router_type
        self.expand = nn.Linear(self.m, 1024, bias=True)  # Expand from m to 1024
        if router_type == "attention":
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                              num_heads=num_heads, 
                                              batch_first=True)
        self.project = nn.Linear(1024, num_experts, bias=True)  # Project 1024 to num_experts
    def forward(self, x):
        x = self.expand(x)  # Expand to 1024
        # x = torch.relu(x)    # Apply non-linearity
        # x= self.hidden(x)  # Hidden layer
        x = F.relu(x)
        if self.router_type == "attention":
            attn_output, _ = self.attn(query=x, key=x, value=x)
            logits = self.project(attn_output)
            return logits
        elif self.router_type == "multi_layer":
            x = self.project(x)  # Project down to num_experts
            return x
        else:
            raise ValueError("Invalid router type")
        
class SharedState:
    sharedgates = None 
    routerloss = 1
    router_loss_weight=None
    router_accuracy = 0

class RouterProjectionUsingBaseLLM(nn.Module):
    def __init__(self, base_classifier, initial_input_ids, router_projection_layer, pad_id_token
                  ):
        super().__init__()
        self.initial_input_ids = initial_input_ids
        self.base_classifier = base_classifier
        self.router_projection_layer = router_projection_layer
        self.pad_token = pad_id_token
        # self.experts_count = experts_count
        
    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        logits = self.router_projection_layer(X)
        # print("\nshape of X inside router projection layer", X.shape)
        # print(X[0][-1][:5], X[0][-2][:5])
        # print("\nshape of outputs from router projection layer", logits.shape)
        # print("\n pad id token", self.pad_token)
        '''Code from transfomers libraray's modelingllama.py'''
        if self.pad_token is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.pad_token is None:
            sequence_lengths = -1
        else:
            if self.initial_input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(self.initial_input_ids, self.pad_token).int().argmax(-1) - 1
                # print("\nshape of sequence_lengths", sequence_lengths.shape)
                sequence_lengths = sequence_lengths % self.initial_input_ids.shape[-1]
                # print("\nshape of sequence_lengths", sequence_lengths.shape)
                sequence_lengths = sequence_lengths.to(logits.device)
                # print("\nshape of sequence_lengths", sequence_lengths.shape)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return pooled_logits

class MoEsparseRoutingForClassification(nn.Module):
     ####################################For Classification Head#########################################
     def __init__(self, base_classifier, config
                 #  router_gates: nn.Module,
                  ):
         super().__init__()
         # self.moe_wrapper = moe_wrapper
         self.base_classifier = base_classifier
         self.config = config
         self.experts = self.config["experts_dict"]
         self.device = self.config["device"]
         # self.broadcast_gates = moe_base_module
         self.num_experts = len(self.experts)
         self.model_name = self.config["model_name"]
         # print("Inside Classification Initialization")
         # print("Gates access", self.broadcast_gates.gates)

    
    
     def custom_classifier_dense_forward(self, X,
                                         selected_weights:torch.Tensor,
                                         selected_biases,
                                         *args, **kwargs):
         X= X.unsqueeze(1)
         if selected_biases is None:
             out = (X * selected_weights).sum(2)
         else:
             out = (X * selected_weights).sum(2) + selected_biases
         # print("output shape", out.shape)
         return out
    
     def custom_classifier_out_proj_forward(self, X,
                                         selected_weights:torch.Tensor,
                                         selected_biases,
                                         *args, **kwargs):
        
         X= X.unsqueeze(1)
         out = (X * selected_weights).sum(2) + selected_biases
         # print("\noutput shape from custom out_proj_forward", out.shape)
         return out
    
     def custom_score_forward(self, X,
                              selected_weights:torch.Tensor,
                                         selected_biases):
         # print("\nshape of X inside classifier projection layer", X.shape)
         # print("\nshape of selected_weights", selected_weights.shape)
         # # X = X.mean(dim=1)
         # print("\nshape of X after mean", X.shape)
         # X= X.unsqueeze(2)
         # weights = selected_weights.unsqueeze(1)
         # print("\nshape of X after unsqueeze", X.shape)
         # out = (X * selected_weights).sum(3)
         out = torch.einsum('bsl,bod->bso', X, selected_weights)
         # print("\nshape of out after sum", out.shape)
         # sys.exit(1)
         return out

     def forward(self, X):
         if "roberta" in self.model_name:
             # Stack classifier dense weights and biases for all experts
             stacked_dense_weights = torch.stack([self.experts[expert]["classifier"]["dense"]["weight"] for expert in self.experts]).to(self.device)
             stacked_dense_biases = torch.stack([self.experts[expert]["classifier"]["dense"]["bias"] for expert in self.experts]).to(self.device)
             # Stack out_proj weights and biases for all experts
             stacked_out_proj_weights = torch.stack([self.experts[expert]["classifier"]["out_proj"]["weight"] for expert in self.experts]).to(self.device)
             stacked_out_proj_biases = torch.stack([self.experts[expert]["classifier"]["out_proj"]["bias"] for expert in self.experts]).to(self.device)

             gates = SharedState.sharedgates.to(self.device)
             B, seq_len, _ = X.shape

             # **Expand gates for proper broadcasting**
             gates_dense = gates.view(B, -1, 1, 1).to(self.device)  # Shape: [B, num_experts, 1, 1] for dense weights
             gates_out_proj = gates.view(B, -1, 1).to(self.device)  # Shape: [B, num_experts, 1] for out_proj biases

             if gates is None:
                 raise ValueError("Gates have not been computed. Ensure the encoder's forward pass runs before classification.")

             # Expand stacked weights to match batch size
             stacked_dense_weights = stacked_dense_weights.unsqueeze(0).to(self.device)  # Shape: [1, 4, 768, 768]
             stacked_dense_biases = stacked_dense_biases.unsqueeze(0).to(self.device)  # Shape: [1, 4, 768]

             stacked_out_proj_weights = stacked_out_proj_weights.unsqueeze(0).to(self.device)  # Shape: [1, 4, 2, 768]
             stacked_out_proj_biases = stacked_out_proj_biases.unsqueeze(0).to(self.device)  # Shape: [1, 4, 2]

             # **Compute dynamically selected classifier weights**
             selected_dense_weights = (stacked_dense_weights * gates_dense).sum(dim=1).to(self.device)  # Shape: [B, 768, 768]
             selected_dense_biases = (stacked_dense_biases * gates_out_proj).sum(dim=1).to(self.device)  # Shape: [B, 768]
             # print("dtype of selected_dense_biases", type(selected_dense_biases))
             # print("biases", selected_dense_biases)
             selected_out_proj_weights = (stacked_out_proj_weights * gates_out_proj.unsqueeze(-1)).sum(dim=1).to(self.device)  # Shape: [B, 2, 768]
             selected_out_proj_biases = (stacked_out_proj_biases * gates_out_proj).sum(dim=1).to(self.device)  # Shape: [B, 2]

             self.base_classifier.dense.forward = partial(self.custom_classifier_dense_forward,
                                                     selected_weights=selected_dense_weights, 
                                                     selected_biases=selected_dense_biases)

             self.base_classifier.out_proj.forward = partial(self.custom_classifier_out_proj_forward, 
                                                     selected_weights=selected_out_proj_weights,
                                                     selected_biases=selected_out_proj_biases)

             # print("classifier_out of roberta", classifier_out)
             # sys.exit(1)
             return self.base_classifier(X)

         elif "llama" in self.model_name:
            #  '''Previous method'''
            #  gates= SharedState.sharedgates.to(self.device)
            #  stacked_score_weights = torch.stack([self.experts[expert]["score"]["weight"] for expert in self.experts]).to(self.device)
            #  B, seq_len, _ = X.shape
            #  gates_score = gates.view(B, -1, 1).to(self.device)  # Shape: [B, 4, 1] for score
            #  if gates is None:
            #      raise ValueError("Gates have not been computed. Ensure the encoder's forward pass runs before classification.")
            #  stacked_score_weights = stacked_score_weights.unsqueeze(0).to(self.device)  # Shape: [1, 4, 2, 4096]
            #  selected_score_weights = (stacked_score_weights * gates_score.unsqueeze(-1)).sum(dim=1).to(self.device)  # Shape: [B, 2, 4096]
            #  outputs = []
            #  for i in range(B):
            #      inp = X[i]
            #     #  print("\nshape of input before", inp.shape)   
            #      weight = selected_score_weights[i]
            #     #  print("\nshape of selected weight", weight.shape)
            #     #  print("\nshape of base_classifier_weights", self.base_classifier.weight.shape)
            #     #  out = weight.matmul(inp)
            #      out = torch.matmul(inp, weight.transpose(0, 1))
            #     #  out = self.base_classifier(inp)
            #     #  print("\nshape of out", out.shape)
            #      outputs.append(out)
            #     #  print("\nshape of outputs as list", len(outputs))
            #     #  sys.exit(1)
            #  #print("\nshape of outputs as list", len(outputs))
            #  outputs = torch.stack(outputs)
            #  return outputs
         
             '''New Method'''
             gates = SharedState.sharedgates.to(self.device)
             B, E = gates.shape

             # Get index of selected expert per sample: [B]
             selected_expert_indices = torch.argmax(gates, dim=1)

             # Prepare score weights and biases
             selected_score_weights = []
             for i in range(B):
                 expert_idx = selected_expert_indices[i].item()
                 expert_key = list(self.experts.keys())[expert_idx]

                 weight = self.experts[expert_key]["score"]["weight"].to(self.device)  # [C_i, D]
                 selected_score_weights.append(weight)
             max_classes = max(self.experts[expert]["score"]["weight"].shape[0] for expert in self.experts)

             outputs = []
             for i in range(B):
                 inp = X[i]  # [D]
                 weight = selected_score_weights[i]  # [Cᵢ, D]
                 out = torch.matmul(inp, weight.T)  # [Cᵢ]
                 pad_size = max_classes - out.shape[1]
                 if pad_size > 0:
                     out = F.pad(out, (0, pad_size), value=float("-inf"))  # Use -inf if you're applying softmax later
                 outputs.append(out)
             outputs = torch.stack(outputs).to(self.device)  # [B, max_classes]
             return outputs
         
class MoEsparseRouting(nn.Module):
    def __init__(self, 
                 base_module: nn.Module, 
                 config: Dict,
                 pad_id_token: int
                 ):
        super().__init__()                      
        self.configuration =config
        self.base_module = base_module
        self.pad_id_token = pad_id_token #for llm router's classification layer

        self.m_factors_q = config["m_factors_q"]
        self.n_factors_q = config["n_factors_q"]
        self.m_factors_v = config["m_factors_v"]
        self.n_factors_v = config["n_factors_v"]

        self.experts = config["experts_dict"]
        self.common_alpha = config["alpha"]
        self.num_experts = len(self.experts)
        self.device = config["device"]
        self.router_type = config["router_type"]
        self.router_loss_weight = config["router_loss_weight"]
        self.gumbel_temperature = config["gumbel_temperature"]
        self.model_name = config["model_name"]
        self.num_cores_q = len(config["qshape"])
        self.num_cores_v = len(config["vshape"])
        self.topk = config["topk"]
        self.gating_noise = config["gating_noise"]
        self.model_path = config["router_model_path"]
        self.num_labels_for_router = config["num_labels"]
        self.experts_trainable = config["experts_trainable"]
        self.dropout_rate = config.get("expert_dropout", 0.1)  # Default 10% dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        #'''Adapting classification layer with MoE Classification Layer'''
        #self.base_module.score = MoEsparseRoutingForClassification(base_classifier=self.base_module.score, config=self.configuration)

        # Compute m, n for Query
        self.m_q = 1
        for f in self.m_factors_q:
            self.m_q *= f
        self.n_q = 1
        for f in self.n_factors_q:
            self.n_q *= f
        
        # Compute m, n for Value
        self.m_v = 1
        for f in self.m_factors_v:
            self.m_v *= f
        self.n_v = 1
        for f in self.n_factors_v:
            self.n_v *= f

        # For different llm as router rather than leveraging the base_module
        def call_separate_llm_router(self):
            model_path = self.model_path
            num_labels = self.num_labels_for_router
            # print(num_labels)
            router_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, ignore_mismatched_sizes=True)
            try:
                router_model.config.pad_token_id = router_model.config.eos_token_id[0]
            except:
                router_model.config.pad_token_id = router_model.config.eos_token_id
                
            for name, param in router_model.named_parameters():
                if "score"  in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("Router type is llm")
            self.router = SeparateLLMRouterLightningModule(router_model, lr=config["learning_rate"])


        # Define the gating noise layer
        if self.gating_noise:
            self.gate_noise = nn.Linear(self.m_q, self.num_experts, bias=True)

        # Define the router
        if self.router_type == "single_layer":
            self.router = nn.Linear(self.m_q, self.num_experts, bias=True)
            print("Router type is single layer")
        elif self.router_type == "multi_layer" or self.router_type == "attention":
            self.router = MultiStepRouter(input_dim=self.m_q, 
                                          num_experts=self.num_experts, 
                                          router_type=self.router_type)
            print("Router type is multi layer or attention")
        elif self.router_type == "llm":
            # call_separate_llm_router(self)

            '''Save base module's foward function before overriding it'''
            self.forward_functions = {}
            layer_idx = 0
            for layer in self.base_module.model.layers:
                self.forward_functions[f'layer_{layer_idx}_query_{layer_idx}'] = layer.self_attn.q_proj.forward
                self.forward_functions[f'layer_{layer_idx}_value_{layer_idx}'] = layer.self_attn.v_proj.forward
                layer_idx += 1
            input_dim = self.m_q
            self.original_score_layer = self.base_module.score
            # print("Shape of input_dim", input_dim)
            self.router_projection_layer = nn.Linear(input_dim, self.num_experts, bias=False)
            # print("Router_projection layer init",self.router_projection_layer)
            
        '''Adapting classification layer with MoE Classification Layer'''
        self.base_module.score = MoEsparseRoutingForClassification(base_classifier=self.base_module.score, config=self.configuration)


    def custom_query_forward(self, 
                             X, 
                             base_layer_weight, 
                             base_layer_bias, 
                             gates, 
                             stacked_query_cores,
                             *args, **kwargs):

        tt_cores_stacked = stacked_query_cores
        base_layer_out = F.linear(input=X, weight=base_layer_weight, bias=base_layer_bias)
        self.num_m = len(self.m_factors_q)
        self.num_n = len(self.n_factors_q)
        self.num_cores = self.num_m + self.num_n
        if len(tt_cores_stacked) != self.num_cores:
            raise ValueError(f'Expected {self.num_cores} cores, got {len(tt_cores_stacked)}')

        # Figure out E from the shape of the first core
        example_core = tt_cores_stacked[0]
        self.num_experts = example_core.shape[0]  # first dimension => E
        stacked_cores_list = tt_cores_stacked
        ttlora_x_computation = tensorized_multiplication_experts(self, 
                                                                 X, 
                                                                 stacked_cores_list, 
                                                                 self.m_factors_q, 
                                                                 self.n_factors_q, 
                                                                 gates)
        
        # (f) Override the forward function of the query layer
        alpha = self.common_alpha
        out = ttlora_x_computation*alpha
        return out + base_layer_out

    def custom_value_forward(self, 
                             X, 
                             base_layer_weight, 
                             base_layer_bias, 
                             gates, 
                             stacked_value_cores,
                             *args, **kwargs):
        tt_cores_stacked = stacked_value_cores
        base_layer_out = F.linear(input=X, weight=base_layer_weight, bias=base_layer_bias)

        # Count the total TT-cores
        self.num_m = len(self.m_factors_v)
        self.num_n = len(self.n_factors_v)
        self.num_cores = self.num_m + self.num_n
        if len(tt_cores_stacked) != self.num_cores:
            raise ValueError(f'Expected {self.num_cores} cores, got {len(tt_cores_stacked)}')

        # Figure out E from the shape of the first core
        example_core = tt_cores_stacked[0]
        self.num_experts = example_core.shape[0]  # first dimension => E
        # print("\n Value Forward")
        ttlora_x_computation = tensorized_multiplication_experts(self, 
                                                                 X, 
                                                                 tt_cores_stacked, 
                                                                 self.m_factors_v, 
                                                                 self.n_factors_v, 
                                                                 gates)
        
        alpha = self.common_alpha
        out = ttlora_x_computation*alpha
        return out + base_layer_out
    
    def count_labels_of_experts_selection(self, gates):
        expert_key = list(self.experts.keys())
        counts = {key: 0 for key in expert_key}
        for row in gates:
            idx = torch.argmax(row)
            counts[expert_key[idx]] += 1
            
        counts["total"] = sum(counts.values())
        for expert in expert_key:
            counts[expert] = (counts[expert]/counts["total"])*100
        return counts

    def calculate_router_loss_and_accuracy(self, gate_logits, expert_label, gates):
        router_loss = F.cross_entropy(gate_logits, expert_label)
        router_correctness = 0
        # expert_label = expert_label.tolist()
        i=0
        for gate in gates:
            # print("\gate value",gate)
            router_label = torch.argmax(gate)
            if router_label == expert_label[i]:
                router_correctness += 1
            i+=1
        router_accuracy = router_correctness/len(gates)
        SharedState.router_accuracy = router_accuracy
        return router_loss

    def forward(self, input_ids=None, attention_mask=None, labels=None, expert_label=None, *args, **kwargs):
        X= input_ids
        input_ids = input_ids
        attention_mask = attention_mask
        labels = labels
        expert_label = expert_label
        X_temp = input_ids
        X= X.to(self.device)
        gumbel_temperature = self.gumbel_temperature
        # print("\nshape of input_ids in forward", input_ids.shape)
        if "roberta" in self.model_name:
            B = X.size(0)
            X_temp = self.base_module.roberta.embeddings.word_embeddings(X_temp).to(self.device)
            X_temp = X_temp.to(torch.int64)
            pooled_hidden_states = X_temp.float().mean(dim=1)
            router_logits = self.router(pooled_hidden_states)
            gates = F.gumbel_softmax(router_logits, tau=gumbel_temperature, hard=True).to(self.device)
            SharedState.sharedgates = gates
            SharedState.router_loss_weight = self.router_loss_weight
            SharedState.routerloss = self.calculate_router_loss(router_logits, expert_label)
            layer_idx = 0
            for layer in self.base_module.roberta.encoder.layer:
                ##################################################For query######################################
                # (a) Collect query TT-cores for all experts
                list_query_cores = [[] for _ in range(self.num_cores_q)]
                # list_query_cores = [[] for _ in range(len(access_first_expert[f"layer_{layer_idx}"]["query"]))]
                for expert in self.experts.values():
                    for i, (core_name, tensor) in enumerate(expert[f'layer_{layer_idx}']["query"].items()):
                        list_query_cores[i].append(tensor)

                # (b) Convert list of 8[ list of 4[ tensor of (r_i, factor_dim, r_{i+1})] ] ==> list of 8 [ tensor of (E, r_i, factor_dim, r_{i+1})]
                stacked_query_cores = [torch.stack(core_list) for core_list in list_query_cores]
                layer.attention.self.query.forward = partial(self.custom_query_forward, 
                                                            base_layer_weight=layer.attention.self.query.weight, 
                                                            base_layer_bias= layer.attention.self.query.bias,
                                                            gates=gates, 
                                                            stacked_query_cores=stacked_query_cores)            
                ##################################################For Value######################################
                # (a) Collect query TT-cores for all experts
                list_value_cores = [[] for _ in range(self.num_cores_v)]
                for expert in self.experts.values():
                    for i, (core_name, tensor) in enumerate(expert[f'layer_{layer_idx}']["value"].items()):
                        list_value_cores[i].append(tensor)

                # (b) Convert list of 8[ list of 4[ tensor of (r_i, factor_dim, r_{i+1})] ] ==> list of 8 [ tensor of (E, r_i, factor_dim, r_{i+1})]
                stacked_value_cores = [torch.stack(core_list) for core_list in list_value_cores]
                layer.attention.self.value.forward = partial(self.custom_value_forward, 
                                                            base_layer_weight=layer.attention.self.value.weight,
                                                            base_layer_bias= layer.attention.self.value.bias,
                                                            gates=gates, 
                                                            stacked_value_cores=stacked_value_cores
                                                            )            
                
                #increase the layer index
                layer_idx += 1
            
            # Forward through transformer layers
            return self.base_module(input_ids=input_ids,attention_mask=attention_mask, labels=labels)  
        
        elif "llama" in self.model_name:
            # print(self.base_module)
            # sys.exit(1)
            X_temp = self.base_module.model.embed_tokens(X_temp).to(self.device)
            B = X_temp.size(0)
            pooled_hidden_states = X_temp.float().mean(dim=1)
            # print(input_ids[0])
            '''Shazeer noisy gating mechanism'''
            if self.gating_noise:
                '''Getting router logits leveraging the base model and custom projection layer'''
                if self.router_type == "llm":
                    layer_idx = 0
                    for layer in self.base_module.model.layers:
                        layer.self_attn.q_proj.forward = self.forward_functions[f'layer_{layer_idx}_query_{layer_idx}']
                        layer.self_attn.v_proj.forward = self.forward_functions[f'layer_{layer_idx}_value_{layer_idx}']
                        layer_idx += 1
                    self.custom_score = RouterProjectionUsingBaseLLM(base_classifier=self.original_score_layer,
                                                                    initial_input_ids=input_ids, 
                                                                    router_projection_layer=self.router_projection_layer, 
                                                                    pad_id_token = self.pad_id_token)
                    base_model_outputs = self.base_module.model(input_ids=input_ids)
                    hidden_states_for_router = base_model_outputs.last_hidden_state
                    gate_logits = self.custom_score(hidden_states_for_router)
                    # '''Adapting back the base module with original score layer and then adapting the 
                    # forward function with MoE Classification Layer in later stage'''
                    # self.base_module.score = self.original_score_layer
                    pooled_hidden_states_for_router = hidden_states_for_router.float().mean(dim=1)
                    noise_scale = F.softplus(self.gate_noise(pooled_hidden_states_for_router))
                else:
                    gate_logits = self.router(pooled_hidden_states)
                    noise_scale = F.softplus(self.gate_noise(pooled_hidden_states))
                noise = torch.randn_like(noise_scale)
                router_logits = gate_logits + noise * noise_scale
                router_logits_for_router_loss = gate_logits + noise * noise_scale
            else:
                router_logits = self.router(pooled_hidden_states)
                router_logits_for_router_loss= self.router(pooled_hidden_states) 
            '''gumbel'''
            # gates = F.gumbel_softmax(router_logits, tau=gumbel_temperature, hard=True).to(self.device)
            '''softmax'''
            # router_logits_softmax = F.softmax(router_logits, dim=1).to(self.device)
            # gates = router_logits_softmax
            '''Shazeer gating mechanism'''
            # Get top-2 indices
            topk_values, topk_indices = torch.topk(router_logits, k=self.topk, dim=1)
            for b in range(B):
                for i in range(router_logits.size(1)):
                    if i not in topk_indices[b]:
                        router_logits[b, i] = float('-inf')

            gates = F.softmax(router_logits, dim=1).to(self.device)
            #print("\n Gates\n", gates[:1])
            #print("\n Router Projection Layer weights\n", self.router_projection_layer.weight[:1])
            SharedState.sharedgates = gates
            SharedState.router_loss_weight = self.router_loss_weight
            SharedState.routerloss = self.calculate_router_loss_and_accuracy(router_logits_for_router_loss, expert_label,gates)
            print(f'\nSelected Experts of this Batch size {B}\n', 
              self.count_labels_of_experts_selection(gates))
            layer_idx = 0
            for layer in self.base_module.model.layers:
                ##################################################For query######################################
                # (a) Collect query TT-cores for all experts
                list_query_cores = [[] for _ in range(self.num_cores_q)]
                # list_query_cores = [[] for _ in range(len(access_first_expert[f"layer_{layer_idx}"]["query"]))]
                for expert in self.experts.values():
                    for i, (core_name, tensor) in enumerate(expert[f'layer_{layer_idx}']["query"].items()):
                        list_query_cores[i].append(tensor)

                # (b) Convert list of 8[ list of 4[ tensor of (r_i, factor_dim, r_{i+1})] ] ==> list of 8 [ tensor of (E, r_i, factor_dim, r_{i+1})]
                stacked_query_cores = [torch.stack(core_list) for core_list in list_query_cores]
                layer.self_attn.q_proj.forward = partial(self.custom_query_forward, 
                                                            base_layer_weight=layer.self_attn.q_proj.weight,
                                                            base_layer_bias = None, 
                                                            gates=gates, 
                                                            stacked_query_cores=stacked_query_cores)            
                ##################################################For Value######################################
                # (a) Collect query TT-cores for all experts
                list_value_cores = [[] for _ in range(self.num_cores_v)]
                for expert in self.experts.values():
                    for i, (core_name, tensor) in enumerate(expert[f'layer_{layer_idx}']["value"].items()):
                        list_value_cores[i].append(tensor)

                # (b) Convert list of 8[ list of 4[ tensor of (r_i, factor_dim, r_{i+1})] ] ==> list of 8 [ tensor of (E, r_i, factor_dim, r_{i+1})]
                stacked_value_cores = [torch.stack(core_list) for core_list in list_value_cores]
                layer.self_attn.v_proj.forward = partial(self.custom_value_forward, 
                                                            base_layer_weight=layer.self_attn.v_proj.weight,
                                                            base_layer_bias= None,
                                                            gates=gates, 
                                                            stacked_value_cores=stacked_value_cores)            
                
                #increase the layer index
                layer_idx += 1
            return self.base_module(input_ids=input_ids,attention_mask=attention_mask, labels=labels)
