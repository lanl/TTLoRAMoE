import pytorch_lightning as pl
import pandas as pd
import torch
import tensorly as tl
import time
import os
from tqdm import tqdm
import warnings
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from argparse import ArgumentParser
from utils import get_tokenizer, load_dataset_,preprocess_datasets
from utils import load_new_model_for_sequence_classification_from_local_path
from TTLoRAWrapper_utils import wrap_model_with_ttcores_contraction, wrap_model_with_ttcores_with_reconstruction

tl.set_backend('pytorch')
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM']='true'
torch.set_float32_matmul_precision('medium')
custom_cache_dir ="./transformers_cache/"
os.environ['HF_DATASETS_CACHE'] = custom_cache_dir
tl.set_backend('pytorch')

def test_inference(config):

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this notebook.")
    dataset = load_dataset_(config["dataset_name"], config["dataset_path"]) #load dataset from local directory
    dataset = preprocess_datasets(config["dataset_name"], dataset) #pre-process datasets like changing labels and removing columns
    sub_dataset = dataset["validation"].select(range(args.batchsize*10)) #only process 10 batches
    tokenized = get_tokenizer(config, sub_dataset)

    val_loader = DataLoader(
        dataset=tokenized,
        batch_size=args.batchsize, 
        num_workers=args.workers,  
        shuffle=False,
    )
    if(args.test=="contraction"):
        ttlora_model = load_new_model_for_sequence_classification_from_local_path(config)
        model = wrap_model_with_ttcores_contraction(ttlora_model, config)
    elif(args.test=="reconstruction"):
        ttlora_reconstruction_model=load_new_model_for_sequence_classification_from_local_path(config)
        model = wrap_model_with_ttcores_with_reconstruction(ttlora_reconstruction_model, config)
    else:
        raise ValueError(f'{args.test} is not a valid option for --test (use either contraction or reconstruction)')
    
    model=model.half()
    model.to(device)
    model.eval()

    batch_times = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            if i == 10:
                break  # Only process 10 batches

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            end = time.time()
            batch_times.append(end - start)

    avg_time = sum(batch_times) / len(batch_times)
    
    if(args.test=="contraction"):
        results = {
            "Method": "TT-LoRA Contraction",
            "Inference Time": avg_time,
            "Batch size": args.batchsize,    
            "For loop":len(batch_times)}
        
        df = pd.DataFrame(list(results.items()), columns=['metric', 'value'])   
        print(df)
        # filename = f'contraction_{config["dataset_name"]}_{args.batchsize}.csv'
        # df.to_csv(f'././Inference_Testing_Results/{filename}', index=True)

    elif(args.test=="reconstruction"):
        results = {
            "Method": "TT-LoRA Reconstruction",
            "Inference Time": avg_time,
            "Batch size": args.batchsize,    
            "For loop":len(batch_times)}
        
        df = pd.DataFrame(list(results.items()), columns=['metric', 'value'])   
        print(df)
        # filename = f'reconstruction_{config["dataset_name"]}_{args.batchsize}.csv'
        # df.to_csv(f'./Inference_Testing_Results/{filename}', index=True)

    else:
        raise ValueError(f'{args.test} is not a valid option for --test (use either contraction or reconstruction)')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="qnli")
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--workers",type=int, default=8)
    parser.add_argument("--test", type=str)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        ##TT-LoRA parameters 
        "rank": 5,
        "alpha": 16,
        "core_init_choice": "init_and_decompose",
        #query parameters 
        "qshape": [16,8,4,4,4,4,8,16], #2048 x 2048
        "m_factors_q": [16,8,4,4], #2048
        "n_factors_q": [16,8,4,4], #2048
        #value parameters
        "vshape": [16,16,4,2,2,16,16],  #2048 x 512
        "m_factors_v": [16,16,4,2], #2048
        "n_factors_v":[16,16,2], #512
        
        #model parameters
        "model_name" : "llama3.2-1b",
        "model_path" : './llama3.2-1b', #provide the model path
        "tokenizer_path" : './llama3.2-1b', #provide the tokenizer path
        "dataset_path": './datasets', #provide the dataset path
        "device": device,
        "dataset_name" : args.dataset, 
    }

    analysis =  test_inference(config)

