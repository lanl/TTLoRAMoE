import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import sys
import warnings
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from single_training_lightning_model import CustomLightningModule
from utils import get_tokenizer, load_dataset_
from utils import load_new_model_for_sequence_classification_from_local_path, preprocess_datasets
from TTLoRAWrapper_utils import wrap_model_with_ttcores_contraction

tl.set_backend('pytorch')
os.environ['TOKENIZERS_PARALLELISM']='true'
torch.set_float32_matmul_precision('medium')
tl.set_backend('pytorch')
warnings.filterwarnings("ignore")
custom_cache_dir ="./transformers_cache/"
os.environ['HF_DATASETS_CACHE'] = custom_cache_dir

torch.cuda.reset_peak_memory_stats()

def train_without_ray(config):

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this notebook.")

    dataset = load_dataset_(config["dataset_name"], config["dataset_path"])
    dataset = preprocess_datasets(config["dataset_name"], dataset)
    tokenized = get_tokenizer(config, dataset)
    train_dataset = tokenized["train"]
    val_dataset = tokenized["validation"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,  # 32 for llama, 256 for roberta
        shuffle=True,   #data shuffles at the beginning of each epoch
        num_workers=args.workers   #16 for llama, 8 for roberta
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batchsize,
        num_workers=args.workers
        #no need to shuffle the validation data as to get the consistent evaluations
    )

    '''Load the model and and define the labels and makes the parameters untrainable'''
    model = load_new_model_for_sequence_classification_from_local_path(config)
    wrapped_model = wrap_model_with_ttcores_contraction(model, config)

    lightning_model = CustomLightningModule(wrapped_model, config)
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
        )
    
    model_checkpoint_callback=ModelCheckpoint(
        dirpath=f'./TTLoRA_Saved_Individual_Expert/{config["dataset_name"]}',
        save_top_k=1, 
        mode="max", 
        monitor="val_acc")  

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        devices=args.gpus,
        log_every_n_steps=10,
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

    print("-"*50, 
          "\nTraining Accuracy: ", train_acc, 
          "\nValidation Accuracy in best lightning model: ", val_acc)

    print("Best model path: ", model_checkpoint_callback.best_model_path)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_params = count_parameters(model)

    results = {"Model_name": config["model_name"],
            "Dataset": config["dataset_name"],
            "Total_epochs": trainer.current_epoch + 1,
            "Trainable_parameters_count": train_params,
            "Query Shape": config["qshape"],
            "Query m_factors": config["m_factors_q"],
            "Query n_factors": config["n_factors_q"],
            "Value Shape": config["vshape"],
            "Value m_factors": config["m_factors_v"],
            "Value n_factors": config["n_factors_v"],
            "Rank": config["rank"],
            "Alpha": config["alpha"],
            "Learning_rate": config["learning_rate"],
            "Taining_Accuracy": train_acc[0]['accuracy'], 
            "Validation_Accuray": val_acc[0]['accuracy'],
            "Training_Time" : training_time,
            "Best_model_path": model_checkpoint_callback.best_model_path,
            }
    
    df = pd.DataFrame(list(results.items()), columns=['metric', 'value'])   
    print(df)
    filename = f"TTLoRA_{dataset_name}.csv"
    df.to_csv(f'./TTLoRA_Experts_Results/{filename}', index=True)
    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()
    
    #changeable model parameter
    model_name = "llama-3.2-1b" 
    dataset_name = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        #ttlora parameters
        #query parameters
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

        
        "rank": 5,
        "alpha": 16, 

        #model parameters
        "model_name" : model_name,
        "model_path" : './llama3.2-1b', #provide the model path
        "tokenizer_path" :'./llama3.2-1b', #provide the tokenizer path
        "dataset_path": "./datasets", #provide the dataset path
        "device": device, 
        "core_init_choice": "direct_init", # options: "direct_init", "init_and_decompose"
    
        "dataset_name" : dataset_name, 
        
        #changeable hyperparameters
        "learning_rate": 5e-3, 
    }

    analysis =  train_without_ray(config)
    
