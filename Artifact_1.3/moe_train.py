import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import os
import math
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial
from datasets import  Dataset

#imports from local files
from moe_lightning_model import CustomLightningModule
from utils import get_tokenizer, parse_experts, parse_experts_make_trainable
from utils import load_new_model_for_moe_sequence_classification_from_local_path, load_dataset_, load_mixed_datasets_with_expertlabels, preprocess_datasets
from utils import load_new_model_for_sequence_classification_from_local_path
from moe_wrapper import MoEsparseRouting 

os.environ['TOKENIZERS_PARALLELISM']='true'
torch.set_float32_matmul_precision('medium')
tl.set_backend('pytorch')
custom_cache_dir = "./transformers_cache/"
os.environ['HF_DATASETS_CACHE'] = custom_cache_dir
os.environ['HF_HOME'] = custom_cache_dir

def train_moe_without_ray(config):

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")

    '''Dataset loading and check if loaded correctly'''
    if config["dataload_type"] == "single":
        model = load_new_model_for_sequence_classification_from_local_path(config)
        dataset = load_dataset_(config["dataset_name"], config["dataset_path"])
        dataset = preprocess_datasets(config["dataset_name"], dataset)
        tokenized = get_tokenizer(config, dataset)
        task_label = config['multiple_datasets'].index(config['dataset_name'])
        train_dataset = tokenized["train"]
        train_dataset = train_dataset.add_column("expert_label", [task_label] * train_dataset.num_rows)
        val_dataset = tokenized["validation"]
        val_dataset = val_dataset.add_column("expert_label", [task_label] * val_dataset.num_rows)

    elif config["dataload_type"] == "multiple":
        #For multiple datasets
        model = load_new_model_for_moe_sequence_classification_from_local_path(config)
        train_dataset_dict, val_dataset_dict = load_mixed_datasets_with_expertlabels(config)
        train_dataset = Dataset.from_dict(train_dataset_dict)
        val_dataset = Dataset.from_dict(val_dataset_dict)
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label", "expert_label"])
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label", "expert_label"])
        train_dataset = train_dataset.shuffle(seed=42)
        val_dataset = val_dataset.shuffle(seed=42)
    else:
        raise ValueError("Please provide the correct dataload type")

    if "llama" in config['model_name']:
        model = MoEsparseRouting(model, config, model.config.pad_token_id)
    else:
        raise ValueError("Please provide the correct model name")
    
    '''Dataloader (an iterable) handles number of rows in each batch and how many gpus to use'''
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True, 
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.workers,
       )

    '''For trainig and evaluation'''
    lightning_model = CustomLightningModule(model, 
                                            config)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
        )

    if config['dataload_type'] == "single":
        dirpath = f"./MoE_Checkpoints/single_{config['dataset_name']}"
    elif config["dataload_type"] == "multiple":
        dirpath = f"./MoE_Checkpoints/multiple_{len(config['multiple_datasets'])}"
    else:
        raise ValueError("Please provide the correct dataload type")

    '''Callback provided by PyTorch Lightning that allows to save model checkpoints during training'''
    model_checkpoint_callback=ModelCheckpoint(
        dirpath=dirpath,
        save_top_k=1, 
        mode="max", 
        monitor="val_acc")  

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        devices=args.gpus,

    )
    
    start = time.time()
    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                )
    end = time.time()
    training_time = end - start
    print(f'Time elapsed {training_time/60:.2f} min')

    '''Evaluating the model on training and validation datasets'''
    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc = trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_params = count_parameters(model)

    results = {"Model_name": config['model_name'],
            "Router Type": config["router_type"],
            "Trainable_parameters_count": train_params,
            "epochs": trainer.current_epoch,
            "router_loss_weight" :config["router_loss_weight"],
            "Experts trainable": config["experts_trainable"],
            "Experts random initialization": config["experts_random_initialization"],
            "topk" :config["topk"],
            "Gating Noise" : config["gating_noise"],
            "data_mix_type":config["datamix_type"],
            "learning_rate": config["learning_rate"],
            "Taining_Accuracy": train_acc[0]['accuracy'], 
            "Training Router Loss": train_acc[0]["router_loss"],
            "Training Total Loss":train_acc[0]['total_loss'],
            "Training Router Acuracy": train_acc[0]['router_acc'],
            "Validation_Accuray": val_acc[0]['accuracy'],
            "Validation Router Loss": val_acc[0]["router_loss"],
            "Validation Total Loss":val_acc[0]['total_loss'],
            "Validation Router Acuracy": val_acc[0]['router_acc'],
            "Experts": config['experts_list'],
            "Query m_factors": config["m_factors_q"],
            "Query n_factors": config["n_factors_q"],
            "Value m_factors": config["m_factors_v"],
            "Value n_factors": config["n_factors_v"],
            "Best_model_path": model_checkpoint_callback.best_model_path,
            "dataload_type": config["dataload_type"],
            "dataset_name" : config["dataset_name"],
            "Training Time": training_time,
            }
    df = pd.DataFrame(list(results.items()), columns=['metric', 'value'])
    if config["dataload_type"] == "single":  
        filename = f"Single_{(config['dataset_name'])}.csv"
    if config["dataload_type"] == "multiple": 
        filename = f"Multiple_{len(config['multiple_datasets'])}.csv"

    df.to_csv(f'./MoE_Training_Results/{filename}', index=False)
    print(df)
    return results
            
def main():
    model_name = "llama-3.2-1b" 

    dataload_type= "single" # {single, multiple}
    dataset_name = args.dataset 
    '''mixed datasets'''
    multiple_datasets= ["mrpc", "cola", "sst2", "qnli", "qqp", "imdb",
                        "hellaswag", "scitail","sick", "mnli"] # combination of the datasets
    
    experts_list= ["mrpc", "cola", "sst2", "qnli", 
                        "rte", "qqp", "imdb","winogrande_l",
                        "hellaswag", "socialiqa", "cosmosqa",
                        "scitail", "csqa", "sick", "cb", "boolq", "mnli"] #combination of experts
    
    # multiple_datasets= ["mrpc", "cola", "rte"]
    # experts_list= ["mrpc", "cola", "rte"]


    experts_trainable = False
    experts_random_initialization = False  
    if experts_trainable:
        experts_dict = parse_experts_make_trainable(f'./TTLoRA_Saved_Individual_Expert/', experts_random_initialization, model_name, dataload_type, dataset_name, experts_list)
    else:
        experts_dict = parse_experts(f'./TTLoRA_Saved_Individual_Expert/', model_name, experts_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #check conditions
    if not experts_dict:
        raise ValueError("The experts dictionary is empty. Please provide valid experts.")


    config = {
        #ttlora parameters
        "qshape": [64,4,3,3,4,64] if "roberta-base" in model_name #roberta query shape = 768x768
        else [16,8,4,4,4,4,8,16] if "llama-3.2-1b" in model_name #llama-3.2-1b q_proj shape = 2048x2048
        else [16,4,4,3,2,2,2,2,3,4,4,16] if "llama-3.2-3b" in model_name #llama-3.2-3b q_proj shape = 3072x3072
        else [16,4,4,4,2,2,2,2,4,4,4,16] if "llama-3.1-8b" in model_name #llama-3.1-8b q_proj shape = 4096x4096,
        else [16,4,4,4,2,2,2,2,2,2,4,4,4,16] if "llama-3.1-70b" in model_name #llama-3.1-70b q_proj shape = 8192x8192
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        "m_factors_q": 
        [64,4,3] if "roberta-base" in model_name #roberta m of query shape = 768,
        else [16,8,4,4] if "llama-3.2-1b" in model_name #llama-3.2-1b m of q_proj shape = 2048
        else [16,4,4,3,2,2] if "llama-3.2-3b" in model_name #llama-3.2-3b m of q_proj shape = 3072
        else [16,4,4,4,2,2] if "llama-3.1-8b" in model_name #llama-3.1-8b m of q_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3.1-70b" in model_name #llama-3.1-70b m of q_proj shape = 8192
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        "n_factors_q": 
        [64,4,3] if "roberta-base" in model_name #roberta n of query shape = 768
        else [16,8,4,4] if "llama-3.2-1b" in model_name #llama-3.2-1b n of q_proj shape = 2048
        else [16,4,4,3,2,2] if "llama-3.2-3b" in model_name #llama-3.2-3b n of q_proj shape = 3072,
        else [16,4,4,4,2,2] if "llama-3.1-8b" in model_name #llama-3.1-8b n of q_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3.1-70b" in model_name #llama-3.1-70b n of q_proj shape = 8192
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        #value parameters
        "vshape": [64,4,3,3,4,64] if "roberta-base" in model_name #roberta value shape = 768x768
        else [16,16,4,2,2,16,16] if "llama-3.2-1b" in model_name #llama-3.2-1b v_proj shape = 2048 x 512
        else [16,4,4,3,2,2,2,2,4,4,16] if "llama-3.2-3b" in model_name #llama-3.2-3b v_proj shape = 3072x1024
        else [16,4,4,4,2,2,2,2,4,4,16] if "llama-3.1-8b" in model_name #llama-3.1-8b v_proj shape = 4096x1024,
        else [16,4,4,4,2,2,2,2,2,4,4,16] if "llama-3.1-70b" in model_name #llama-3.1-70b v_proj shape = 8192x1024
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        "m_factors_v": 
        [64,4,3] if "roberta-base" in model_name #roberta m of value shape = 768
        else [16,16,4,2] if "llama-3.2-1b" in model_name #llama-3.2-1b m of v_proj shape = 2048
        else [16,4,4,3,2,2] if "llama-3.2-3b" in model_name #llama-3.2-3b m of v_proj shape = 3072,
        else [16,4,4,4,2,2] if "llama-3.1-8b" in model_name #llama-3.1-8b m of v_proj shape = 4096,
        else [16,4,4,4,2,2,2] if "llama-3.1-70b" in model_name #llama-3.1-70b m of v_proj shape = 8192
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        "n_factors_v": 
        [64,4,3] if "roberta-base" in model_name #roberta n of value shape = 768
        else [16,16,2] if "llama-3.2-1b" in model_name #llama-3.2-1b m of v_proj shape = 512
        else [16,4,4,2,2] if "llama-3.2-3b" in model_name #llama-3-3b n of v_proj shape = 1024
        else [16,4,4,2,2] if "llama-3.1-8b" in model_name #llama-3.1-8b n of v_proj shape = 1024
        else [16,4,4,2,2] if "llama-3.1-70b" in model_name #llama-3.1-70b n of v_proj shape = 1024
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        
        "rank": 
        5 if "roberta-base" in model_name 
        else 5 if "llama" in model_name
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        "alpha": 
        16 if "roberta-base" in model_name 
        else 16 if "llama" in model_name
        else (lambda: (_ for _ in ()).throw(ValueError(f'{model_name} Not adapted for this experiment')))(),

        #model parameters
        "model_name" : model_name,
        "device": device, 
        "model_path" : './llama3.2-1b', #provide the model path
        "tokenizer_path" :'./llama3.2-1b', #provide the tokenizer path
        "dataset_path": "./datasets", #provide the dataset path
  
        #changable dataset parameters:
        "dataload_type": dataload_type,
        "dataset_name" : dataset_name, 
        "multiple_datasets": multiple_datasets,
        "num_labels": len(experts_list),
        "experts_list" : experts_list,

        #experts and moe parameters
        "experts_dict": experts_dict,
        "experts_trainable": experts_trainable,
        "expert_dropout": 0.1,
        "gating_noise": True,
        "gumbel_temperature": 1.0,
        
        #hyperparameters
        "topk" : 1,
        "router_type" : args.router, # {single_layer, multi_layer, attention,llm}
        "router_model_path" : '/dummypath/', #only if router_type is loaded using different llm other than base model. 
        "learning_rate": 1e-3,
        "router_loss_weight": 21, #scalar value
        "datamix_type": "balanced", # {balanced, unbalanced}
        "experts_random_initialization" : experts_random_initialization,
        "experts_trainable" : experts_trainable
    }

    analysis =  train_moe_without_ray(config)
    
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--router", type=str, default="llm")
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main()
