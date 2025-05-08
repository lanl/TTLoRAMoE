import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import warnings
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from single_training_lightning_model import CustomLightningModule
from utils import get_tokenizer, load_dataset_, preprocess_datasets
from utils import load_new_model_for_sequence_classification_from_local_path
from LoRAWrapper_utils import wrap_model_with_lora

tl.set_backend('pytorch')
os.environ['TOKENIZERS_PARALLELISM']='true'
custom_cache_dir ="./transformers_cache/"
os.environ['HF_DATASETS_CACHE'] = custom_cache_dir
torch.set_float32_matmul_precision('medium')
tl.set_backend('pytorch')
warnings.filterwarnings("ignore")

def train_lora(config):

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this notebook.")

    dataset = load_dataset_(config["dataset_name"], config["dataset_path"])
    dataset = preprocess_datasets(config["dataset_name"], dataset)
    tokenized = get_tokenizer(config, dataset)
    train_dataset = tokenized["train"]
    val_dataset = tokenized["validation"]

    '''Dataloader (an iterable) handles number of rows in each batch and how many gpus to use'''
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,  
        shuffle=True,  
        num_workers=args.workers  
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batchsize,
        num_workers=args.workers
    )

    '''Load the model and and define the labels and makes the parameters untrainable'''
    model = load_new_model_for_sequence_classification_from_local_path(config)

    wrapped_model = wrap_model_with_lora(model, config)
    lightning_model = CustomLightningModule(wrapped_model, config)
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
        )
    
    model_checkpoint_callback=ModelCheckpoint(
        dirpath=f'./LoRA_Saved_Individual_Expert/{config["dataset_name"]}/',
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

    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc = trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_params = count_parameters(model)
    
    results = {
        "Model_name": config["model_name"],
        "Dataset": config["dataset_name"],
        "Total_epochs": trainer.current_epoch + 1,
        "Trainable_parameters_count": train_params,
        "Rank": config["lora_rank"],
        "Alpha": config["lora_alpha"],
        "Learning_rate": config["learning_rate"],
        "Taining_Accuracy": train_acc[0]['accuracy'], 
        "Validation_Accuray": val_acc[0]['accuracy'],
        "Best_model_path": model_checkpoint_callback.best_model_path,
            }
    
    df = pd.DataFrame(list(results.items()), columns=['metric', 'value'])   
    print(df)
    filename = f"LoRA_{dataset_name}.csv"
    df.to_csv(f'./LoRA_Experts_Results/{filename}', index=True)
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
    model_name = "llama3.2-1b" 
    dataset_name = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        #LoRA hyperparameters obtained from ray tuning
        "lora_rank": 16,
        "lora_alpha": 8,
        "learning_rate": 0.0005,
        #model parameters
        "model_name" : model_name,
        "model_path" : './llama3.2-1b', #provide the model path
        "tokenizer_path" :'./llama3.2-1b', #provide the tokenizer path
        "dataset_path": "./datasets", #provide the dataset path
        "device": device, 
        "dataset_name" : dataset_name,

        
    }
    analysis =  train_lora(config)