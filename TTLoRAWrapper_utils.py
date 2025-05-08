import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import tensorly as tl
import math, sys
import tensorly as tl
import torch.nn.init as init
from tensorly.decomposition import tensor_train
from functools import partial

tl.set_backend('pytorch') #used to set the backend for the tensorly library to PyTorch

def tensorized_multiplication_old(X, tt_cores, m_factors, n_factors, device):
    # for i in range(1):
    #print("\nInput ids inside tensorized multiplication\n", X[0][0][:5])
    # sys.exit(1)
    B = X.size(0)
    S_len = X.size(1)

    # 1) Reshape X => [B, m1, m2, ...]
    shape_x = [B]+[S_len] + m_factors[::-1]
    tt_state = X.view(shape_x)  # e.g. [B, seq_len, m1, m2,....] if m_factors=[4,3]
    # print("tt_state shape intial",tt_state.shape)
    # print(tt_state[0][0][0][0][0][:5])
    # 2) Insert an initial rank dimension => [B, 1, m1, m2, ...]
    tt_state = tt_state.unsqueeze(1)  # shape [B, r=1, m1, m2, ...]
    # print("tt_state expanded",tt_state.shape)
    # print(tt_state[0][0][0][0][0][0][:5])
    # We'll do:
    #   - first "len(m_factors)" cores:  contract out each input factor
    #   - next "len(n_factors)" cores:   add each output factor dimension
    num_m = len(m_factors)
    num_n = len(n_factors)
    total_cores = num_m + num_n

    if len(tt_cores) != total_cores:
        raise ValueError(f'Expected {total_cores} TT-cores, got {len(tt_cores)}')

    # 3) Process each INPUT factor
    # We want to remove the last dimension from tt_state each time:
    # eq = 'b r ... m, r m p -> b p ...'
    # Explanation:
    #   - 'm' is the last dimension
    #   - '...' lumps any leftover dims in between 'r' and 'm'
    #   - we sum over (r, m), leaving 'b' and leftover '...' plus new rank p
    for i in range(num_m):
       
        core = tt_cores[i]  # shape [r_in, m_i, r_out]
        #print("\ncore shape",core.shape)
        #print(f"\ncore {i} in m factors\n",core[0][0][:5])
        # base_core = layer_weight_cores[i]
        # core = base_core + alpha * initial_core
        # print(core.shape,tt_state.shape)
        eq_in = 'b r ... m, r m p -> b p ...'
        tt_state = torch.einsum(eq_in, tt_state, core)
        # print("tt_state shape after each contraction",tt_state.shape)
        
        # shape now has same # of dims, except the "m" dimension is contracted out
        # and rank dimension might have changed from r_in to r_out

    # 4) Process each OUTPUT factor
    # Now each output factor is appended at the end:
    # eq = 'b r ..., r n p -> b p ... n'
    # Explanation:
    #   - we sum over 'r'
    #   - leftover dims remain in '...'
    #   - new factor dimension 'n' is appended at the end
    # print("tt_state shape after m factors contraction\n",tt_state[0][0][:5])
    start_n = num_m
    for i in range(num_n):
        core = tt_cores[start_n + i]  # shape [r_in, n_i, r_out]
        #print("\ncore shape",core.shape)
        #print(f"\ncore {i} in n factors\n",core[0][0][:5])
        # base_core = layer_weight_cores[start_n + i]
        # core = base_core + alpha * initial_core
        eq_out = 'b r ..., r n p -> b p ... n'
        tt_state = torch.einsum(eq_out, tt_state, core)
        # print("tt_state shape",tt_state.shape)
        # shape now has one more dimension at the end for 'n'

    # print("tt_state shape after n factors contraction\n",tt_state[0][0][0][0][0][0][:5])
    # 5) Flatten to [B, -1] => [B, prod(n_factors)]
    Z = tt_state.view(B,S_len, -1) #[B, seq_len, hidden_dim]
    #print("Z Output from tensor multiplication\n",Z[0][0][:5])
    return Z

def tensorized_multiplication(X, tt_cores, m_factors, n_factors, device):
    B = X.size(0)
    S_len = X.size(1)
    
    # Ensure inputs are on the correct device (avoid implicit transfers)
    X = X.to(device, non_blocking=True)
    tt_cores = [core.to(device, non_blocking=True) for core in tt_cores]
    
    # Validate core count
    num_m = len(m_factors)
    num_n = len(n_factors)
    total_cores = num_m + num_n
    
    if len(tt_cores) != total_cores:
        raise ValueError(f'Expected {total_cores} TT-cores, got {len(tt_cores)}')
    
    # 1 & 2) Reshape and add rank dimension efficiently
    # Ensure tensor is contiguous for better memory access patterns
    if not X.is_contiguous():
        X = X.contiguous()
    
    # Combine reshape and unsqueeze in one efficient operation
    shape_x = [B] + [S_len] + m_factors[::-1]
    tt_state = X.view(shape_x).unsqueeze(1)
    
    # 3) Process INPUT factors efficiently
    for i in range(num_m):
        core = tt_cores[i]
        tt_state = torch.einsum('br...m,rmp->bp...', tt_state, core)
    
    # 4) Process OUTPUT factors efficiently
    for i in range(num_n):
        core = tt_cores[num_m + i]
        tt_state = torch.einsum('br...,rnp->bp...n', tt_state, core)
    
    # 5) Efficiently reshape to final form
    Z = tt_state.view(B, S_len, -1)
    
    return Z

def get_ttlora_shape(ttlora_shape_from_config):
    ttlora_shape = ttlora_shape_from_config
    return ttlora_shape

def get_ttlora_rank(r, ttlora_shape):
    ttlora_rank = [1]
    for i in range(len(ttlora_shape)-1):
        ttlora_rank.append(r)
    ttlora_rank.append(1)
    return ttlora_rank

class TTLoRALinearWrapper_Contraction(nn.Module): 
        '''Define cores and the forward pass and makes it ready for training the cores'''
        def __init__(self, module: nn.Module, tt_shape, tt_rank, alpha:int, m_factors, n_factors, device, init_choice):
            super().__init__()
            self.base_module = module
            self.tt_shape = tt_shape
            self.tt_rank = tt_rank
            self.alpha=alpha
            self.m_factors = m_factors
            self.n_factors = n_factors
            self.device = device
            self.init_choice = init_choice
            self.in_features_shape, self.out_features_shape = self.base_module.weight.shape
            
            if self.init_choice == "direct_init":
                self.tt_cores = self.generate_cores(self.tt_shape, self.tt_rank).to(self.device)  # Change method as needed
                self.tt_cores.requires_grad= True 
                # Make the bias non-trainable
                if self.base_module.bias is not None:
                        self.base_module.bias.requires_grad = False
            
            elif self.init_choice == "init_and_decompose":
                '''Create a torch tensor dummy Weight_delta of shape (in_feature_shape, out_feature_shape) 
                and initialize all 0s'''
                self.Weight_delta=torch.zeros((self.in_features_shape, self.out_features_shape)).to('cuda')
                '''Then allocate random values using gaussian distribution to dummy Weight_delta'''
                self.reset_parameters()
                '''Decompose the dummy Weight_delta to high dimensional tensor based on the TT shapes'''
                self.Weight_TT_dimension = self.reshape_tensor(torch.tensor(self.Weight_delta)).to('cuda')
                '''We have dummy weight decomposed into multiple tensors based on tt_shape
                Now, we create tensor cores as Parameters which are trainable
                Paramerter wraps the tensors into traninable parameters
                ParameterList holds the list of parameters
                TT Cores are initialized using standard normal distribution based on the ttcores shapes'''
                self.tt_cores = nn.ParameterList([nn.Parameter(self.initialize_cores(*shape).to('cuda')) for shape in self.get_ttcores_shapes()])
                '''Using tensor train, decompose into multiple tensors based on the ranks and shapes provided'''
                self.tt_cores_dummy = tensor_train(self.Weight_TT_dimension, self.tt_rank)
                '''Transfer the values of tensor trained ttlora_cores_dummy to ttlora_cores trainable parameters'''
                for i in range(len(self.tt_cores)):
                    self.tt_cores[i].data = torch.tensor(self.tt_cores_dummy[i], dtype=torch.float32).to('cuda')
            
                self.tt_cores.requires_grad= True 
                # Make the bias non-trainable
                if self.base_module.bias is not None:
                        self.base_module.bias.requires_grad = False
            else:
                raise ValueError("Invalid initialization choice")

        def generate_cores(self, shape, rank):
            tt_cores = nn.ParameterList()  # Store TT cores as trainable parameters

            for i in range(len(shape)):
                core_shape = (rank[i], shape[i], rank[i + 1])  # TT core shape
                core = torch.empty(core_shape)  # Create empty tensor
            
                tt_cores.append(nn.Parameter(core))  # Store as a trainable parameter
            
            for i in range(len(tt_cores)):
                    nn.init.kaiming_uniform_(tt_cores[i], a=math.sqrt(8))
                    tt_cores[i].data /= (tt_cores[i].data.norm() + 1e-6)  # Normalize cores

            return tt_cores 
        
        def get_ttcores_shapes(self):
            shapes = []
            ranks = self.tt_rank
            for i in range(len(self.tt_shape)):
                shape = (ranks[i], self.tt_shape[i], ranks[i + 1])
                shapes.append(shape)
            return shapes

        def reshape_tensor(self, tensor):
            return tensor.reshape(*self.tt_shape) ## * unpacks the tt_shape list into individual arguments

        def reset_parameters(self):
            '''Initialize the given tensor with random values from a gaussian distribution'''
            torch.manual_seed(42)
            nn.init.kaiming_uniform_(self.Weight_delta, a=math.sqrt(8))
        
        def reset_parameters(self):
            '''Initialize the given tensor with random values from a gaussian distribution'''
            torch.manual_seed(42)
            nn.init.kaiming_uniform_(self.Weight_delta, a=math.sqrt(8))

        def initialize_cores(self, *shape):
            '''Initialize the given tensor with random values from a standard normal distribution (mean = 0 and std = 1)
            and scaled by a calculated standard deviation'''
            std = 1.0 / math.sqrt(shape[1]) #Standard deviation
            return torch.randn(*shape) * std
            
        def forward(self, x: torch.Tensor) -> torch.Tensor: # x is input used in forward pass at every call of model
            if self.alpha > 0:
                out = tensorized_multiplication(x.to(self.device), 
                                                self.tt_cores, 
                                                m_factors=self.m_factors, 
                                                n_factors=self.n_factors, 
                                                device=self.device) 

                return self.base_module(x.to(self.device)) + out*self.alpha

class TTLoRALinearWrapper_Reconstruction(nn.Module):
        
        def __init__(self, module: nn.Module, tt_shape, tt_rank, alpha:int):
            super().__init__()

            self.base_module = module
            self.in_features, self.out_features = self.base_module.weight.shape
            self.tt_shape = tt_shape
            self.alpha=alpha
            self.tt_rank = tt_rank
            self.tt_shape = tt_shape
            self.W_delta=torch.zeros((self.in_features, self.out_features)) ### create a torch matrix W_delta of shape (in_feature, out_feature) and initialize iit to all 0s
            self.reset_parameters() ### this function will basically allocate random valuesa from gaussian distribution to W_delta

            #make 10d torch
            self.W_10d = self.reshape_to_10d(torch.tensor(self.W_delta)) ### decompose the W_delta to high dimentional tensor based on the TT shapes

            #tt cores
            ### create model parameters. The paramaters will be multiple tensors, where the shape of each tensor is determined by provided ranks and TT shapes.
            ### Now, these tensors will be initialized randomly. Later, we'll transfer the values of W_delta to these paramaters
            self.tt_cores = nn.ParameterList([nn.Parameter(self.init_core(*shape)) for shape in self.get_tt_shapes()])
            # tl.set_backend('pytorch')

            ### using tensor train, decompose the W_delta into multiple tensors based on the ranks and shapes provided
            self.tt_cores_dummy = tensor_train(self.W_10d, self.tt_rank)

            ### transfer the values of tt_cores_dummy to self.tt_cores which are the newly added parameters of the model
            for i in range(len(self.tt_cores)):
                self.tt_cores[i].data = torch.tensor(self.tt_cores_dummy[i], dtype=torch.float32)

            self.tt_cores.requires_grad= True ### make self.tt_cores trainable

            # self.base_module.weight.requires_grad = False ### make base_module's parameters non-trainable

            ### MAKE THE BIAS NON-TRAINABL
            if self.base_module.bias is not None:
                    self.base_module.bias.requires_grad = False
            # self.reset_parameters()

        def get_tt_shapes(self):
            shapes = []
            ranks = self.tt_rank
            for i in range(len(self.tt_shape)):
                shape = (ranks[i], self.tt_shape[i], ranks[i + 1])
                shapes.append(shape)
            return shapes

        def reshape_to_10d(self, tensor):
            return tensor.reshape(*self.tt_shape)

        def reset_parameters(self):

            torch.manual_seed(42)
            nn.init.kaiming_uniform_(self.W_delta, a=math.sqrt(8))
            # nn.init.kaiming_uniform_(self.tt_cores_up, a=math.sqrt(5))

        def init_core(self, *shape):
            std = 1.0 / math.sqrt(shape[1])
            return torch.randn(*shape) * std

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.alpha > 0:
                ### need to multiply all the tensors to gvet the original shape of the high dimensional tensor (the tensor before decomposition)
                if len(self.tt_shape) == 4:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq->jlnp', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3])
                if len(self.tt_shape) == 6:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq,qrs,stu->jlnprt', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3], self.tt_cores[4], self.tt_cores[5])
                if len(self.tt_shape) == 7:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw->jlnprtv', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3], self.tt_cores[4], self.tt_cores[5], self.tt_cores[6])
                if len(self.tt_shape) == 8:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw,wxy->jlnprtvx', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3], self.tt_cores[4], self.tt_cores[5], self.tt_cores[6], self.tt_cores[7])
                if len(self.tt_shape) == 10:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw,wxy,yza,abc->jlnprtvxzb', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3], self.tt_cores[4], self.tt_cores[5], self.tt_cores[6], self.tt_cores[7], self.tt_cores[8], self.tt_cores[9])
                if len(self.tt_shape) == 12:
                    self.tt_weights = torch.einsum('ijk,klm,mno,opq,qrs,stu,uvw,wxy,yza,abc,cde,efg->jlnprtvxzbdf', self.tt_cores[0], self.tt_cores[1], self.tt_cores[2], self.tt_cores[3], self.tt_cores[4], self.tt_cores[5], self.tt_cores[6], self.tt_cores[7], self.tt_cores[8], self.tt_cores[9], self.tt_cores[10], self.tt_cores[11])

                adapted_weight = self.base_module.weight + self.alpha * (self.tt_weights.reshape(self.in_features, self.out_features))
                return F.linear(x, adapted_weight, self.base_module.bias)
            else:
                return F.linear(x, self.base_module.weight, self.base_module.bias)
            
def wrap_model_with_ttcores_contraction(model, config):
    ttlora_shape_q = get_ttlora_shape(config["qshape"])
    ttlora_rank_q = get_ttlora_rank(config["rank"], ttlora_shape_q)

    ttlora_shape_v = get_ttlora_shape(config["vshape"])
    ttlora_rank_v = get_ttlora_rank(config["rank"], ttlora_shape_v)

    m_factors_q = config["m_factors_q"]
    n_factors_q = config["n_factors_q"]
    m_factors_v = config["m_factors_v"]
    n_factors_v = config["n_factors_v"]

    ttlora_alpha = config["alpha"]
    ttlora_adapter_at_query = True
    ttlora_adapter_at_value = True

    assign_ttlora = partial(TTLoRALinearWrapper_Contraction, alpha=ttlora_alpha)
    
    if "llama" in config["model_name"]:
        for layer in model.model.layers:
            if ttlora_adapter_at_query:
                layer.self_attn.q_proj = assign_ttlora(layer.self_attn.q_proj,
                                                       tt_shape=ttlora_shape_q, 
                                                       tt_rank=ttlora_rank_q,
                                                       m_factors=m_factors_q,
                                                       n_factors=n_factors_q,
                                                       device=config["device"],
                                                       init_choice=config["core_init_choice"])
            if ttlora_adapter_at_value:
                layer.self_attn.v_proj = assign_ttlora(layer.self_attn.v_proj,
                                                       tt_shape=ttlora_shape_v,
                                                       tt_rank=ttlora_rank_v,
                                                       m_factors=m_factors_v,
                                                       n_factors=n_factors_v,
                                                       device=config["device"],
                                                       init_choice=config["core_init_choice"])
    else:
        raise ValueError("Model name not recognized. Please use 'llama' in the model name.")
    return model

def wrap_model_with_ttcores_with_reconstruction(model, config):
    
    '''Define the shape,rank and other configuration of the tensor train decomposition'''
    ttlora_shape_q = get_ttlora_shape(config["qshape"])
    ttlora_shape_v = get_ttlora_shape(config["vshape"])
    ttlora_rank_q = get_ttlora_rank(config["rank"], ttlora_shape_q)
    ttlora_rank_v = get_ttlora_rank(config["rank"], ttlora_shape_v)
    ttlora_alpha = config["alpha"]
    
    '''Define where to adapt the ttlora'''
    ttlora_adapter_at_query = True
    ttlora_adapter_at_value = True
    
    assign_ttlora_q = partial(TTLoRALinearWrapper_Reconstruction, tt_shape=ttlora_shape_q, tt_rank=ttlora_rank_q, alpha=ttlora_alpha)
    assign_ttlora_v = partial(TTLoRALinearWrapper_Reconstruction, tt_shape=ttlora_shape_v, tt_rank=ttlora_rank_v, alpha=ttlora_alpha)
    if "llama" in config["model_name"]:
        for layer in model.model.layers:
            if ttlora_adapter_at_query:
                layer.self_attn.q_proj = assign_ttlora_q(layer.self_attn.q_proj)
            if ttlora_adapter_at_value:
                layer.self_attn.v_proj = assign_ttlora_v(layer.self_attn.v_proj)

    else:
        raise ValueError("Model name not recognized. Please use 'roberta' or 'llama' in the model name.")
    return model
