import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from adapters import AutoAdapterModel

from single_training_lightning_model import CustomLightningModule
from utils import load_new_model_for_sequence_classification_from_local_path, load_dataset_, get_tokenizer, preprocess_datasets

os.environ['TOKENIZERS_PARALLELISM']='true'
torch.set_float32_matmul_precision('medium')
tl.set_backend('pytorch')
custom_cache_dir = "./transformers_cache/"
os.environ['HF_DATASETS_CACHE'] = custom_cache_dir
os.environ['HF_HOME'] = custom_cache_dir

torch.cuda.reset_peak_memory_stats()

def train_moe_without_ray(config):
    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")

    '''Load the model and and define the labels'''
    model = load_new_model_for_sequence_classification_from_local_path(config)

    dataset = load_dataset_(config["dataset_name"], config["dataset_path"])
    dataset = preprocess_datasets(config["dataset_name"], dataset)
    tokenized = get_tokenizer(config, dataset)
    train_dataset = tokenized["train"]
    val_dataset = tokenized["validation"]

    '''Adapt model with adapters'''
    model_config = model.config
    adapted_model = AutoAdapterModel.from_pretrained(
        config["model_path"],
        config=model_config,
    )

    data = config["dataset_name"]
    if data == "socialiqa" or data =="sick" or data == "cb"in data or data == "mnli":
        num_classes = 3
    elif data == "cosmosqa" or data == "hellaswag":
        num_classes = 4
    elif data == "csqa":
        num_classes = 5
    else:
        num_classes = 2

    adapted_model.add_adapter(f"{config['dataset_name']}_seq_bn", config="seq_bn")
    adapted_model.train_adapter(f"{config['dataset_name']}_seq_bn")
    adapted_model.add_classification_head(f"{config['dataset_name']}_seq_bn", num_labels=num_classes)

    '''Dataloader (an iterable) handles number of rows in each batch and how many gpus to use'''
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,  # 32 for llama, 256 for roberta
        shuffle=True,   #data shuffles at the beginning of each epoch
        num_workers=args.workers, #16 for llama, 8 for roberta
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.workers,
       )

    '''For trainig and evaluation'''
    lightning_model = CustomLightningModule(adapted_model, 
                                            config)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
        )

    '''Callback provided by PyTorch Lightning that allows to save model checkpoints during training'''
    model_checkpoint_callback=ModelCheckpoint(
        dirpath=f'./Adapter_Saved_Individual_Expert/Adapter_Checkpoints/{config["dataset_name"]}/',
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

    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc = trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)

    adapted_model.save_adapter(f'./Adapter_Saved_Individual_Expert/{config["dataset_name"]}_seq_bn', f'{config["dataset_name"]}_seq_bn', with_head=True)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_params = count_parameters(adapted_model)
    print("-"*50, "\nTraining Accuracy: ", train_acc, "\nValidation Accuracy: ", val_acc, "\n Trainable Parameters: ", train_params)


    results = {"Model_name": config["model_name"],
            "Dataset": config["dataset_name"],
            "Total_epochs": trainer.current_epoch + 1,
            "Trainable_parameters_count": train_params,
            "Learning_rate": config["learning_rate"],
            "Taining_Accuracy": train_acc[0]['accuracy'], 
            "Validation_Accuray": val_acc[0]['accuracy'],
            "Training_Time" : training_time,
            "Best_model_path": model_checkpoint_callback.best_model_path,
            }
    
    df = pd.DataFrame(list(results.items()), columns=['metric', 'value'])   
    print(df)
    filename = f"Adapter_{config['dataset_name']}.csv"
    df.to_csv(f'./Adapter_Experts_Results/{filename}', index=True)
    return results

            
def main():
    dataset_name = args.dataset
    model_name = "llama3.2-1b" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        #model parameters
        "model_name" : model_name,
        "device": device, 
        "model_path" : './llama3.2-1b', #provide the model path
        "tokenizer_path" :'./llama3.2-1b', #provide the tokenizer path
        "dataset_path": "./datasets", #provide the dataset path
        "dataset_name": dataset_name, 
        "learning_rate": 1e-4,
    }

    analysis =  train_moe_without_ray(config)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main()
