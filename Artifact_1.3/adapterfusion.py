import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import numpy as np
import time
import os
import math
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, EvalPrediction
from adapters import AdapterTrainer
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial
from datasets import load_dataset, Dataset
from torch import nn
import adapters
from adapters import AutoAdapterModel, list_adapters, get_adapter_info, list_adapters
from adapters.composition import Fuse

from fusion_lightning_model import CustomLightningModule
from utils import load_dataset_, get_tokenizer, get_mix_tokenizer, preprocess_datasets, load_mixed_datasets

os.environ['TOKENIZERS_PARALLELISM']='true'
torch.set_float32_matmul_precision('medium')
tl.set_backend('pytorch')
custom_cache_dir ="./transformers_cache/"
os.environ['HF_DATASETS_CACHE'] = custom_cache_dir  
os.environ['HF_HOME'] = custom_cache_dir


def train_moe_without_ray(config):
    
    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")


    if config["dataload_type"] == "single":
        dataset = load_dataset_(config["dataset_name"], config["dataset_path"])
        dataset = preprocess_datasets(config["dataset_name"], dataset)
        tokenized = get_tokenizer(config, dataset)
        train_dataset = tokenized["train"]
        val_dataset = tokenized["validation"]

    elif config["dataload_type"] == "multiple":
        #For multiple datasets
        train_dataset_dict, val_dataset_dict = load_mixed_datasets(config["model_name"], config["multiple_datasets"], config["tokenizer_path"], config["dataset_path"])
        train_dataset = Dataset.from_dict(train_dataset_dict)
        val_dataset = Dataset.from_dict(val_dataset_dict)
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        train_dataset = train_dataset.shuffle(seed=42)
        val_dataset = val_dataset.shuffle(seed=42)

    else:
        raise ValueError("Please provide the correct dataload type")

    model = AutoAdapterModel.from_pretrained(config["model_path"])
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/mrpc_seq_bn', load_as = "mrpc", with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/cola_seq_bn', load_as = "cola",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/rte_seq_bn', load_as = "rte",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/sst2_seq_bn', load_as = "sst2",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/qnli_seq_bn', load_as = "qnli",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/qqp_seq_bn', load_as = "qqp",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/imdb_seq_bn', load_as = "imdb",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/winogrande_l_seq_bn', load_as = "winogrande_l",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/hellaswag_seq_bn', load_as = "hellaswag",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/socialiqa_seq_bn', load_as = "socialiqa",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/cosmosqa_seq_bn', load_as = "cosmosqa",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/scitail_seq_bn', load_as = "scitail",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/csqa_seq_bn', load_as = "csqa",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/sick_seq_bn', load_as = "sick",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/cb_seq_bn', load_as = "cb",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/boolq_seq_bn', load_as = "boolq",with_head = False)
    model.load_adapter(f'./Adapter_Saved_Individual_Expert/mnli_seq_bn', load_as = "mnli",with_head = False)
    fusion_of=["mrpc", "cola", "sst2", "qnli", 
                        "rte", "qqp", "imdb","winogrande_l",
                        "hellaswag", "socialiqa", "cosmosqa",
                        "scitail", "csqa", "sick", "cb", "boolq"]
    adapter_setup = Fuse("mrpc", "cola", "sst2", "qnli", 
                        "rte", "qqp", "imdb","winogrande_l",
                        "hellaswag", "socialiqa", "cosmosqa",
                        "scitail", "csqa", "sick", "cb", "boolq")
    # fusion_of=["mrpc", "cola", "rte"]
    # adapter_setup = Fuse("mrpc", "cola", "rte")
    model.add_adapter_fusion(adapter_setup)

    data = config["dataset_name"]
    if config["dataload_type"] == "single":
        if data == "socialiqa" or data =="sick" or data == "cb"in data or data == "mnli":
            num_classes = 3
        elif data == "cosmosqa" or data == "hellaswag":
            num_classes = 4
        elif data == "csqa":
            num_classes = 5
        else:
            num_classes = 2
    if config["dataload_type"] == "multiple":
        num_classes = 5

    model.add_classification_head(f'{config["dataset_name"]}', num_labels=num_classes)
    model.train_adapter_fusion(adapter_setup)
    
    if config["dataload_type"]=="single":
        output_dir = f"./AdapterFusion_Training_Results/{config['dataset_name']}"
    elif config["dataload_type"] == "multiple":
        output_dir = f"./AdapterFusion_Training_Results/{len(config['multiple_datasets'])}"
    training_args = TrainingArguments(
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=50,
        output_dir=output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=False,
    )

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        acc= (preds == p.label_ids).mean()
        print("Accuracy", acc)
        return {"acc":acc}


    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_accuracy,
    )
    start = time.time()
    train_results = trainer.train()
    end = time.time()
    training_time = end - start
    print(f'Time elapsed {training_time/60:.2f} min')

    results = trainer.evaluate()

    def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_params = count_parameters(model)
    print("-"*50,"\nValidation Accuracy: ", results['eval_acc'], "\n Trainable Parameters: ", train_params)
    return{ "Model_name": config['model_name'],
            "Taining_Runtime_old": train_results.metrics["train_runtime"], 
            "Training_Time": training_time,
            "Validation_Accuray": results['eval_acc'],
            "Trainable_parameters_count": train_params,
            "epochs": train_results.metrics['epoch'],
            "learning_rate": config["learning_rate"],
            "fusion_of": fusion_of,
            }

def main():
    dataset_name = args.dataset
    model_name = "llama-3.2-1b" # options: roberta-base, llama-3.2-1b, llama-3.2-3b, llama-3.1-8b, llama-3.1-70b, 
    dataload_type= "single" # {single, multiple} 
    multiple_datasets= ["mrpc", "cola", "sst2", "qnli", "qqp", "imdb",
                        "hellaswag", "scitail","sick", "mnli"]
    # multiple_datasets= ["mrpc", "rte","cola"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        #model parameters
        "model_name" : model_name,
        "device": device, 
        "model_path" : './llama3.2-1b', #provide the model path
        "tokenizer_path" :'./llama3.2-1b', #provide the tokenizer path
        "dataset_path": "./datasets", #provide the dataset path
        "dataload_type": dataload_type,
        "dataset_name": dataset_name,
        "multiple_datasets":multiple_datasets, 
        "learning_rate": 5e-5,
    }

    analysis =  train_moe_without_ray(config)
    df = pd.DataFrame(list(analysis.items()), columns=['metric', 'value'])
    print(df)
    if config["dataload_type"] == "single":
        filename = f"adapterfusion_multiple_{config['dataset_name']}.csv"
    elif config["dataload_type"] == "multiple":   
        filename = f"adapterfusion_multiple_{len(config['multiple_datasets'])}.csv"
    df.to_csv(f'./AdapterFusion_Training_Results/{filename}', index=True)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main()