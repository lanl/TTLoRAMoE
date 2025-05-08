import os
from datasets import load_dataset, Dataset, ClassLabel, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F 
import math, sys
from collections import defaultdict
import tensorly as tl

def load_dataset_(data_name, path):
    data_path = os.path.join(path, data_name)
    dataset = load_dataset(data_path)
    return dataset

def preprocess_datasets(dataset_name, dataset):
    '''for datasets like winogrande to convert the labels from 1,2 to 0,1'''
    def remap_labels(example):
                label = str(example["label"]).strip()  # Just in case there's whitespace
                if label == "":
                    example["label"] = -1  # or any default value you prefer
                elif label == "neutral" or label == "NEUTRAL":
                    example["label"] = 0
                elif label == "entailment" or label == "ENTAILMENT":
                    example["label"] = 1
                elif label == "contradiction" or label == "CONTRADICTION":
                    example["label"] = 2
                elif label == "A":
                    example["label"] = 0
                elif label == "B":
                    example["label"] = 1
                elif label == "C":
                    example["label"] = 2
                elif label == "D":
                    example["label"] = 3
                elif label == "E":
                    example["label"] = 4
                else:
                    example["label"] = int(label) - 1
                return example
    
    def flatten_choices(example):
        example["choices"] = example["choices"]["text"]
        return example
    
    def restructure_endings(example):
                for i, choice in enumerate(example["endings"]):
                    example[f"endings{i}"] = choice
                del example["endings"]
                return example

    def flatten_hellaswag(dataset):
        all_texts = []
        all_endings = []
        all_labels = []
        for j, example in enumerate(dataset):
            ctx = f"{example['activity_label']}: {example['ctx_a']} {example['ctx_b']} {example['ctx']}"
            endings = f"{example['endings0']} {example['endings1']} {example['endings2']} {example['endings3']}"
            all_texts.append(ctx)
            all_endings.append(endings)
            all_labels.append(int(example["label"]))
        return {"text": all_texts, "ending":all_endings, "label": all_labels}
    
    def flatten_cosmosqa(dataset):
        all_context_questions = []
        all_answers = []
        all_labels = []
        for j, example in enumerate(dataset):
            context_question = f"{example['context']} {example['question']}"
            answers = f"option0: {example['answer0']}, option1: {example['answer1']}, option2: {example['answer2']}, option3: {example['answer3']}"
            all_context_questions.append(context_question)
            all_answers.append(answers)
            all_labels.append(int(example["label"]))
        return {"context_question": all_context_questions, "answers": all_answers, "label": all_labels}
    
    def flatten_socialiqa(dataset):
        all_context_questions = []
        all_answers = []
        all_labels = []
        for j, example in enumerate(dataset):
            context_question = f"{example['context']} {example['question']}"
            answers = f"option0: {example['answerA']}, option1: {example['answerB']}, option2: {example['answerC']}"
            all_context_questions.append(context_question)
            all_answers.append(answers)
            all_labels.append(int(example["label"]))
        return {"context_question": all_context_questions, "answers": all_answers, "label": all_labels}

    def restructure_choices(example):
            for i, choice in enumerate(example["choices"]):
                example[f"choice{i}"] = choice
            del example["choices"]
            return example

    def flatten_scitail(dataset):
        all_sentence1 = []
        all_sentence2 = []
        all_labels = []
        for j, example in enumerate(dataset):
            sentence1 = f"{example['sentence1_binary_parse']} {example['sentence1_parse']}  {example['sentence1']}"
            sentence2 = f"{example['sentence2_parse']} {example['sentence2']}"
            all_sentence1.append(sentence1)
            all_sentence2.append(sentence2)
            all_labels.append(int(example["label"]))
        return {"sentence1": all_sentence1, "sentence2":all_sentence2, "label": all_labels}
    
    def flatten_csqa(dataset):
        all_question = []
        all_choices = []
        all_question_concept = []
        all_labels = []
        for j, example in enumerate(dataset):
            question = f"{example['question']}"
            choices = f"{example['choice0']}, {example['choice1']}, {example['choice2']}, {example['choice3']}, {example['choice4']}"
            question_concept = f"{example['question_concept']}"
            all_question.append(question)
            all_choices.append(choices)
            all_question_concept.append(question_concept)
            all_labels.append(int(example["label"]))
        return {"question": all_question, "question_concept":all_question_concept, "choices": all_choices, "label": all_labels}
    
    '''change answer column into label column and labels as 0,1 instead of 1,2 for consistency'''
    if "winogrande" in dataset_name:
        dataset = dataset.rename_column("answer", "label")
        dataset = dataset.map(remap_labels)
    if "hellaswag" in dataset_name:
        columns_to_remove = ["ind", "source_id", "split", "split_type"]
        dataset = dataset.remove_columns(columns_to_remove)
        dataset = dataset.map(restructure_endings)
        flattened_train = flatten_hellaswag(dataset["train"])
        flattened_validation = flatten_hellaswag(dataset["validation"])
        flattened_dataset_train = Dataset.from_pandas(pd.DataFrame(flattened_train))
        flattened_dataset_validation = Dataset.from_pandas(pd.DataFrame(flattened_validation))
        dataset = DatasetDict({
                        "train": flattened_dataset_train,
                        "validation": flattened_dataset_validation,
                    })
    if "socialiqa" in dataset_name:
        dataset = dataset.map(remap_labels)
        flattened_train = flatten_socialiqa(dataset["train"])
        flattened_validation = flatten_socialiqa(dataset["validation"])
        flattened_dataset_train = Dataset.from_pandas(pd.DataFrame(flattened_train))
        flattened_dataset_validation = Dataset.from_pandas(pd.DataFrame(flattened_validation))
        dataset = DatasetDict({
                        "train": flattened_dataset_train,
                        "validation": flattened_dataset_validation,
                    })
    if "cosmosqa" in dataset_name:
        flattened_train = flatten_cosmosqa(dataset["train"])
        flattened_validation = flatten_cosmosqa(dataset["validation"])
        flattened_dataset_train = Dataset.from_pandas(pd.DataFrame(flattened_train))
        flattened_dataset_validation = Dataset.from_pandas(pd.DataFrame(flattened_validation))
        dataset = DatasetDict({
                        "train": flattened_dataset_train,
                        "validation": flattened_dataset_validation,
                    })
    if "scitail" in dataset_name:
        columns_to_remove = ["annotator_labels"]
        dataset = dataset.remove_columns(columns_to_remove)
        dataset = dataset.rename_column("gold_label", "label")
        dataset = dataset.map(remap_labels)
        flattened_train = flatten_scitail(dataset["train"])
        flattened_validation = flatten_scitail(dataset["validation"])
        flattened_dataset_train = Dataset.from_pandas(pd.DataFrame(flattened_train))
        flattened_dataset_validation = Dataset.from_pandas(pd.DataFrame(flattened_validation))
        dataset = DatasetDict({
                        "train": flattened_dataset_train,
                        "validation": flattened_dataset_validation,
                    })
    if "csqa" in dataset_name:
        dataset = dataset.rename_column("answerKey", "label")
        dataset = dataset.map(remap_labels)
        dataset = dataset.map(flatten_choices)
        dataset = dataset.map(restructure_choices)
        flattened_train = flatten_csqa(dataset["train"])
        flattened_validation = flatten_csqa(dataset["validation"])
        flattened_dataset_train = Dataset.from_pandas(pd.DataFrame(flattened_train))
        flattened_dataset_validation = Dataset.from_pandas(pd.DataFrame(flattened_validation))
        dataset = DatasetDict({
                        "train": flattened_dataset_train,
                        "validation": flattened_dataset_validation,
                    })
    if dataset_name == "cb":
        columns_to_remove = ['idx', 'template_name', 'template', 'rendered_input', 'rendered_output']
        dataset = dataset.remove_columns(columns_to_remove)

    if dataset_name == "sick":
        columns_to_remove = ['pair_ID', 'relatedness_score', 'entailment_AB', 'entailment_BA', 'sentence_A_original', 'sentence_B_original', 'sentence_A_dataset', 'sentence_B_dataset', 'SemEval_set']
        dataset = dataset.remove_columns(columns_to_remove)
        dataset = dataset.rename_column("entailment_label", "label")
        dataset = dataset.map(remap_labels)

    if dataset_name == "boolq":
        dataset = dataset.rename_column("answer", "label")
        df_train = dataset["train"].to_pandas()
        df_validation = dataset["validation"].to_pandas()
        df_train["label"] = df_train["label"].astype(int)
        df_validation["label"] = df_validation["label"].astype(int)
        train_dataset = Dataset.from_pandas(df_train)
        validation_dataset = Dataset.from_pandas(df_validation)
        dataset = DatasetDict({"train": train_dataset, "validation":validation_dataset})
    return dataset

def get_tokenizer(config, dataset):
    '''Tokenizes the provided dataset and data name using the tokenizer from the specified path'''
    path = config["tokenizer_path"]
    data_name = config["dataset_name"]
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_text(batch):
        if data_name == "sst2":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "mrpc":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "cola":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "qnli":
            return tokenizer(batch["question"], batch['sentence'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "rte":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "qqp":
            return tokenizer(batch["question1"], batch['question2'], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "imdb":
            return tokenizer(batch["text"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "winogrande_l":
            return tokenizer(batch["sentence"], batch["option1"],batch["option2"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "hellaswag":
            return tokenizer(batch["text"], batch["ending"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "socialiqa" or data_name == "cosmosqa" :
            return tokenizer(batch["context_question"], batch["answers"], add_special_tokens=True, truncation=True, padding=True)
        if  data_name == "scitail":
            return tokenizer(batch["sentence1"], batch["sentence2"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "csqa":
            return tokenizer(batch["question"], batch["question_concept"], batch["choices"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "cb" or data_name == "mnli":
            return tokenizer(batch["premise"], batch["hypothesis"], add_special_tokens=True, truncation=True, padding=True)
        if  data_name == "sick":
            return tokenizer(batch["sentence_A"], batch["sentence_B"], add_special_tokens=True, truncation=True, padding=True)
        if data_name == "boolq":
            return tokenizer(batch["question"], batch["passage"],  truncation=True, padding=True)

    tokenized = dataset.map(tokenize_text, batched=True, batch_size=None) 

    ### change the format into tensors of the specific columns
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized

def load_new_model_for_sequence_classification_from_local_path(config):
    if "socialiqa" in config["dataset_name"] or "sick" in config["dataset_name"] or "cb" in config["dataset_name"] or "mnli" in config["dataset_name"]:
        model = AutoModelForSequenceClassification.from_pretrained(config["model_path"], num_labels=3)
    elif config["dataset_name"] == "cosmosqa" or config["dataset_name"] == "hellaswag":
        model = AutoModelForSequenceClassification.from_pretrained(config["model_path"], num_labels=4)
    elif config["dataset_name"] == "csqa":
        model = AutoModelForSequenceClassification.from_pretrained(config["model_path"], num_labels=5)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(config["model_path"], num_labels=2)
    try:
        model.config.pad_token_id = model.config.eos_token_id[0]
    except:
        model.config.pad_token_id = model.config.eos_token_id
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_new_model_for_moe_sequence_classification_from_local_path(config):
    dataset = config["multiple_datasets"]

    if "csqa" in dataset:
        num_labels = 5
    elif "cosmosqa" in dataset or "hellaswag" in dataset:
        num_labels = 4
    elif "socialiqa" in dataset or "sick" in dataset or "cb" in dataset or "mnli" in dataset:
        num_labels = 3
    else:
        num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(config["model_path"], num_labels=num_labels)
    try:
        model.config.pad_token_id = model.config.eos_token_id[0]
    except:
        model.config.pad_token_id = model.config.eos_token_id
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_mix_tokenizer(model_name, path, data_name, dataset): #used to do the maximum padding for the dataset to match all the types of dataset's sequence_length
    '''Tokenizes the provided dataset and data name using the tokenizer from the specified path'''
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    if "roberta" in model_name:
        max_context_length = 512
    elif "llama" in model_name:
        max_context_length = 1024
    else:
        raise ValueError("Model name not recognized. Please use 'roberta' or 'llama' in the model name.")
    def tokenize_text(batch):
        if data_name == "sst2":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "mrpc":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "cola":
            return tokenizer(batch["sentence"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "qnli":
            return tokenizer(batch["question"], batch['sentence'], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "rte":
            return tokenizer(batch["sentence1"], batch['sentence2'], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "qqp":
            return tokenizer(batch["question1"], batch['question2'], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "imdb":
            return tokenizer(batch["text"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "winogrande_l":
            return tokenizer(batch["sentence"], batch["option1"],batch["option2"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "hellaswag":
            return tokenizer(batch["text"], batch["ending"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "socialiqa" or data_name == "cosmosqa" :
            return tokenizer(batch["context_question"], batch["answers"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if  data_name == "scitail":
            return tokenizer(batch["sentence1"], batch["sentence2"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "csqa":
            return tokenizer(batch["question"], batch["question_concept"], batch["choices"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "cb" or data_name == "mnli":
            return tokenizer(batch["premise"], batch["hypothesis"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if  data_name == "sick":
            return tokenizer(batch["sentence_A"], batch["sentence_B"], add_special_tokens=True, truncation=True, padding="max_length", max_length=max_context_length)
        if data_name == "boolq":
            return tokenizer(batch["question"], batch["passage"],  truncation=True, padding="max_length", max_length=max_context_length)
        
    tokenized = dataset.map(tokenize_text, batched=True, batch_size=None) 
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized

def load_mixed_datasets(model_name, dataset_names, tokenizer_path, dataset_path):
    '''Dataset loading and check if loaded correctly'''
    mixed_train_dataset_dict = {
        
        "input_ids": torch.empty(0,dtype=torch.int64),
        "attention_mask": torch.empty(0, dtype=torch.int64),
        "label": torch.empty(0, dtype=torch.int64),
    }
    mixed_validation_dataset_dict = {
        "input_ids": torch.empty(0, dtype=torch.int64),
        "attention_mask": torch.empty(0, dtype=torch.int64),
        "label": torch.empty(0, dtype=torch.int64),
    }
    for dataset_name in dataset_names:
        take_train = 2490 #for rte 2490, mrpc 3668
        take_val = 277 #for rte 277, mrpc 408
        print("Loading dataset inside mixed datasets: ", dataset_name)
        # if dataset_name in ["mrpc", "cola", "sst2", "qnli", "rte", "qqp"]:
        #     glue_type = "glue"
        # elif dataset_name in ["boolq", "wic"]:
        #     glue_type = "super_glue"


        dataset = load_dataset_(dataset_name, dataset_path)


        tokenized = get_mix_tokenizer(model_name, tokenizer_path, dataset_name , dataset)
        train_tokenized_dataset = tokenized["train"]
        train_tokenized_dataset = train_tokenized_dataset.remove_columns(
            [col for col in train_tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        )
        print("Train tokenized dataset before slicing: ", train_tokenized_dataset['input_ids'].shape, train_tokenized_dataset['attention_mask'].shape, train_tokenized_dataset['label'].shape)

        # print("Train tokenized dataset: ", train_tokenized_dataset['input_ids'].shape, train_tokenized_dataset['attention_mask'].shape, train_tokenized_dataset['label'].shape)
        validation_tokenized_dataset = tokenized["validation"]
        validation_tokenized_dataset = validation_tokenized_dataset.remove_columns(
            [col for col in train_tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        )
        print("Validation tokenized dataset before slicing: ", validation_tokenized_dataset['input_ids'].shape, validation_tokenized_dataset['attention_mask'].shape, validation_tokenized_dataset['label'].shape)

        #########################################For Train###################################################
        mixed_train_dataset_dict["input_ids"] = torch.cat((mixed_train_dataset_dict["input_ids"], 
                                                           train_tokenized_dataset["input_ids"][:take_train]), dim=0)
        mixed_train_dataset_dict["attention_mask"] = torch.cat((mixed_train_dataset_dict["attention_mask"], 
                                                                train_tokenized_dataset["attention_mask"][:take_train]), dim=0)
        mixed_train_dataset_dict["label"] = torch.cat((mixed_train_dataset_dict["label"], 
                                                       train_tokenized_dataset["label"][:take_train]), dim=0)
        #########################################For Validation###################################################

        mixed_validation_dataset_dict["input_ids"] = torch.cat((mixed_validation_dataset_dict["input_ids"], 
                                                                validation_tokenized_dataset["input_ids"][:take_val]), dim=0)
        mixed_validation_dataset_dict["attention_mask"] = torch.cat((mixed_validation_dataset_dict["attention_mask"], 
                                                                     validation_tokenized_dataset["attention_mask"][:take_val]), dim=0)
        mixed_validation_dataset_dict["label"] = torch.cat((mixed_validation_dataset_dict["label"], 
                                                            validation_tokenized_dataset["label"][:take_val]), dim=0)
    
    train_indices = torch.randperm(mixed_train_dataset_dict["input_ids"].size(0))
    mixed_train_dataset_dict["input_ids"] = mixed_train_dataset_dict["input_ids"][train_indices]
    mixed_train_dataset_dict["attention_mask"] = mixed_train_dataset_dict["attention_mask"][train_indices]
    mixed_train_dataset_dict["label"] = mixed_train_dataset_dict["label"][train_indices]

    # Shuffle the validation dataset
    val_indices = torch.randperm(mixed_validation_dataset_dict["input_ids"].size(0))
    mixed_validation_dataset_dict["input_ids"] = mixed_validation_dataset_dict["input_ids"][val_indices]
    mixed_validation_dataset_dict["attention_mask"] = mixed_validation_dataset_dict["attention_mask"][val_indices]
    mixed_validation_dataset_dict["label"] = mixed_validation_dataset_dict["label"][val_indices]
    
    return mixed_train_dataset_dict, mixed_validation_dataset_dict

def parse_experts(directory_path, model_name, experts_list):

    """
    Parses all `.ckpt` files inside expert subfolders and organizes the experts into a nested dictionary.
    Saves ttlora cores and classifier weights for each expert.
    """
    all_experts = defaultdict(lambda: defaultdict(lambda: {"query": {}, "value": {}}))
    expert_names = experts_list
    for expert_name in expert_names:
        expert_folder = os.path.join(directory_path, expert_name)

        # Ensure it is a directory (expert folder)
        if os.path.isdir(expert_folder):
            # Iterate through .ckpt files inside the expert folder
            for filename in os.listdir(expert_folder):
                # Check if there are multiple .ckpt files in the expert folder
                ckpt_files = [f for f in os.listdir(expert_folder) if f.endswith(".ckpt")]
                if len(ckpt_files) > 1:
                    raise ValueError(f'Multiple .ckpt files found in {expert_folder}. Only one .ckpt file is allowed per expert folder.')
                if filename.endswith(".ckpt"):
                    file_path = os.path.join(expert_folder, filename)
                    # Load the .ckpt file
                    checkpoint = torch.load(file_path, map_location="cpu")
                    # Extract model weights (state_dict)
                    expert_data = checkpoint["state_dict"] 
                    if "llama" in model_name:
                        expert_data = {k: v for k, v in expert_data.items() if 'tt_cores' in k or 'score' in k}
                        for full_key, tensor in expert_data.items():
                            tensor.requires_grad = False
                            parts = full_key.split(".")
                            if 'score' in parts:  #key as: model.score.weight (no bias)
                                try:   
                                    classifier = parts[1]
                                    w_b=parts[2]
                                    all_experts[expert_name][classifier][w_b] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')  
                            else:        #model.model.layers.1.self_attn.q_proj.ttlora_cores.7
                                try:
                                    layer = f'layer_{parts[3]}'  # Extract layer index
                                    attention_type = parts[5]  # 'query' or 'value'
                                    if attention_type == "q_proj":
                                        attention_type = "query"
                                    elif attention_type == "v_proj":
                                        attention_type = "value"
                                    ttlora_core = parts[-1]  # 'ttlora_cores.<index>'

                                    # Store extracted weights inside dictionary
                                    all_experts[expert_name][layer][attention_type][f'tt_cores_{ttlora_core}'] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')

    return all_experts

def parse_experts_for_lora(directory_path, model_name, experts_list):
    """
    Parses all `.ckpt` files inside expert subfolders and organizes the experts into a nested dictionary.
    Saves lora and classifier weights for each expert.
    """
    all_experts = defaultdict(lambda: defaultdict(lambda: {"query": {}, "value": {}}))
    expert_names = experts_list
    for expert_name in expert_names:
        expert_folder = os.path.join(directory_path, expert_name)

        # Ensure it is a directory (expert folder)
        if os.path.isdir(expert_folder):
            # Iterate through .ckpt files inside the expert folder
            for filename in os.listdir(expert_folder):
                # Check if there are multiple .ckpt files in the expert folder
                ckpt_files = [f for f in os.listdir(expert_folder) if f.endswith(".ckpt")]
                if len(ckpt_files) > 1:
                    raise ValueError(f'Multiple .ckpt files found in {expert_folder}. Only one .ckpt file is allowed per expert folder.')
                if filename.endswith(".ckpt"):
                    file_path = os.path.join(expert_folder, filename)
                    # Load the .ckpt file
                    checkpoint = torch.load(file_path, map_location="cpu")
                    # Extract model weights (state_dict)
                    expert_data = checkpoint["state_dict"]
                    if "llama" in model_name:
                        expert_data = {k: v for k, v in expert_data.items() if 'lora' in k or 'score' in k}
                        for full_key, tensor in expert_data.items():
                            tensor.requires_grad = False
                            parts = full_key.split(".")
                            if 'score' in parts:  #key as: model.score.weight (no bias)
                                try:   
                                    classifier = parts[1]
                                    w_b=parts[2]
                                    all_experts[expert_name][classifier][w_b] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')  
                            else:        #model.model.layers.1.self_attn.q_proj.ttlora_cores.7
                                try:
                                    layer = f'layer_{parts[3]}'  # Extract layer index
                                    attention_type = parts[5]  # 'query' or 'value'
                                    if attention_type == "q_proj":
                                        attention_type = "query"
                                    elif attention_type == "v_proj":
                                        attention_type = "value"
                                    lora = parts[-2] 
                                    # Store extracted weights inside dictionary
                                    all_experts[expert_name][layer][attention_type][f'{lora}'] = tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')
    return all_experts

def parse_experts_make_trainable(directory_path, experts_random_initialization, model_name, dataload_type, dataset_name, experts_list):

    """
    Parses all `.ckpt` files inside expert subfolders and organizes the experts into a nested dictionary.
    Saves ttlora cores and classifier weights for each expert.
    """
    # Nested dictionary to hold all experts
    all_experts = defaultdict(lambda: defaultdict(lambda: {"query": {}, "value": {}}))
    # print(dataload_type, dataset_name, multiple_datasets)
    # Iterate through each expert folder in the directory
    if dataload_type == "multiple":
        expert_names = experts_list
    elif dataload_type == "single":
        expert_names = dataset_name
    else:
        raise ValueError("Invalid dataload type. Please use 'single' or 'multiple' for dataload type.")
    # print(expert_names)
    for expert_name in expert_names:
        expert_folder = os.path.join(directory_path, expert_name)

        # Ensure it is a directory (expert folder)
        if os.path.isdir(expert_folder):
            # Iterate through .ckpt files inside the expert folder
            for filename in os.listdir(expert_folder):
                # Check if there are multiple .ckpt files in the expert folder
                ckpt_files = [f for f in os.listdir(expert_folder) if f.endswith(".ckpt")]
                if len(ckpt_files) > 1:
                    raise ValueError(f'Multiple .ckpt files found in {expert_folder}. Only one .ckpt file is allowed per expert folder.')
                if filename.endswith(".ckpt"):
                    file_path = os.path.join(expert_folder, filename)
                    # Load the .ckpt file
                    checkpoint = torch.load(file_path, map_location="cpu")
                    # Extract model weights (state_dict)
                    expert_data = checkpoint["state_dict"] 
                    if "llama" in model_name:
                        expert_data = {k: v for k, v in expert_data.items() if 'tt_cores' in k or 'score' in k}
                        for full_key, tensor in expert_data.items():
                            tensor.requires_grad = False
                            parts = full_key.split(".")
                            if 'score' in parts:  #key as: model.score.weight (no bias)
                                try:   
                                    classifier = parts[1]
                                    w_b=parts[2]
                                    param_tensor = torch.nn.Parameter(tensor.clone().detach(), requires_grad=False)
                                    all_experts[expert_name][classifier][w_b] = param_tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')  
                            else:        #model.model.layers.1.self_attn.q_proj.ttlora_cores.7
                                try:
                                    layer = f'layer_{parts[3]}'  # Extract layer index
                                    attention_type = parts[5]  # 'query' or 'value'
                                    if attention_type == "q_proj":
                                        attention_type = "query"
                                    elif attention_type == "v_proj":
                                        attention_type = "value"
                                    ttlora_core = parts[-1]  # 'ttlora_cores.<index>'

                                    # Store extracted weights inside dictionary
                                    if experts_random_initialization:
                                        #print("\n Initializing experts randomly")
                                        torch.manual_seed(42)
                                        param_tensor = torch.nn.Parameter(torch.empty(tensor.shape), requires_grad=True)
                                        nn.init.kaiming_uniform_(param_tensor, a=math.sqrt(8))
                                        param_tensor.data /= (param_tensor.data.norm()+1e-6)
                                    else:
                                        #print("\n Parsing experts from pre-trained weights")
                                        param_tensor = torch.nn.Parameter(tensor.clone().detach(), requires_grad=True)
                                    all_experts[expert_name][layer][attention_type][f'tt_cores_{ttlora_core}'] = param_tensor
                                except IndexError:
                                    print(f'Skipping invalid key: {full_key} in {expert_name}')

    return all_experts

def print_experts(experts):
    for expert_key, expert in experts.items():
        print("\nExpert Key", expert_key)   
        for layer, qv in expert.items():
            print(layer)
            for key, value in qv.items():
                print(key)
                # print("\nKey or Query", key)
                if isinstance(value, dict):
                    # print(value.keys())
                    # print("\nValue", value.keys())
                    for key, tensor in value.items():
                        print(key)
                        # print("\nKey", key)
                        print(tensor.shape)
                        # print(tensor)
                        # break
                else:
                    print(value.shape)
                    # print(value)
                    # break
        #         break
        #     break
        # break     
    return experts

def load_mixed_datasets_with_expertlabels(config):
    '''Dataset loading and check if loaded correctly'''
    mixed_train_dataset_dict = {
        
        "input_ids": torch.empty(0,dtype=torch.int64),
        "attention_mask": torch.empty(0, dtype=torch.int64),
        "label": torch.empty(0, dtype=torch.int64),
        "expert_label": torch.empty(0, dtype=torch.int64)
    }
    mixed_validation_dataset_dict = {
        "input_ids": torch.empty(0, dtype=torch.int64),
        "attention_mask": torch.empty(0, dtype=torch.int64),
        "label": torch.empty(0, dtype=torch.int64),
        "expert_label": torch.empty(0, dtype=torch.int64)
    }
    dataset_names = config["multiple_datasets"]
    tokenizer_path = config["tokenizer_path"]
    model_name = config["model_name"]
    datamix_type = config["datamix_type"] 
    dataset_path = config["dataset_path"]
    expert_keys = list(config["experts_dict"].keys())
    # expert_keys = config["experts_list"]  # Ordered list of keys
    
    
    # Count the number of samples in each dataset
    dataset_counts_train = {}
    dataset_counts_valid = {}
    for dataset in dataset_names:
        dataset_loaded = load_dataset_(dataset, dataset_path)
        dataset_counts_train[dataset] = len(dataset_loaded['train'])
        dataset_counts_valid[dataset] = len(dataset_loaded['validation'])
    # Find the dataset with the lowest count
    lowest_count_train_dataset = min(dataset_counts_train, key=dataset_counts_train.get)
    lowest_count_valid_dataset = min(dataset_counts_valid, key=dataset_counts_valid.get)
    take_train = dataset_counts_train[lowest_count_train_dataset]
    take_val = dataset_counts_valid[lowest_count_valid_dataset]
    
    for dataset_name in dataset_names:
        expert_idx = expert_keys.index(dataset_name)
        print("slicing for train: ", take_train, "slicing for val: ", take_val)
        print("Loading dataset inside mixed datasets: ", dataset_name)
        print("expert_idx"  , expert_idx) 
        dataset = load_dataset_(dataset_name, dataset_path)
        dataset = preprocess_datasets(dataset_name, dataset)
        print(dataset)
        tokenized = get_mix_tokenizer(model_name, tokenizer_path, dataset_name , dataset)
        
        #########################################For Train###################################################
        train_tokenized_dataset = tokenized["train"]
        train_tokenized_dataset = train_tokenized_dataset.remove_columns(
            [col for col in train_tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        )
        # Add expert_label column
        train_tokenized_dataset = train_tokenized_dataset.add_column(name="expert_label", column=[expert_idx] * len(train_tokenized_dataset))
        print("Train tokenized dataset before slicing: ", train_tokenized_dataset['input_ids'].shape, 
              train_tokenized_dataset['attention_mask'].shape, 
              train_tokenized_dataset['label'].shape,
              train_tokenized_dataset['expert_label'].shape)

        if datamix_type == "balanced":
            mixed_train_dataset_dict["input_ids"] = torch.cat((mixed_train_dataset_dict["input_ids"], 
                                                            train_tokenized_dataset["input_ids"][:take_train]), dim=0)
            mixed_train_dataset_dict["attention_mask"] = torch.cat((mixed_train_dataset_dict["attention_mask"], 
                                                                    train_tokenized_dataset["attention_mask"][:take_train]), dim=0)
            mixed_train_dataset_dict["label"] = torch.cat((mixed_train_dataset_dict["label"], 
                                                        train_tokenized_dataset["label"][:take_train]), dim=0)
            mixed_train_dataset_dict["expert_label"] = torch.cat((mixed_train_dataset_dict["expert_label"],
                                                                    train_tokenized_dataset["expert_label"][:take_train]), dim=0)
        elif datamix_type == "unbalanced":
            mixed_train_dataset_dict["input_ids"] = torch.cat((mixed_train_dataset_dict["input_ids"], 
                                                            train_tokenized_dataset["input_ids"]), dim=0)
            mixed_train_dataset_dict["attention_mask"] = torch.cat((mixed_train_dataset_dict["attention_mask"], 
                                                                    train_tokenized_dataset["attention_mask"]), dim=0)
            mixed_train_dataset_dict["label"] = torch.cat((mixed_train_dataset_dict["label"], 
                                                        train_tokenized_dataset["label"]), dim=0)
            mixed_train_dataset_dict["expert_label"] = torch.cat((mixed_train_dataset_dict["expert_label"],
                                                                    train_tokenized_dataset["expert_label"]), dim=0)
        else:
            raise ValueError("Invalid datamix type. Please use 'balanced' or 'unbalanced' for datamix type.")
        #########################################For Validation###################################################
        validation_tokenized_dataset = tokenized["validation"]
        validation_tokenized_dataset = validation_tokenized_dataset.remove_columns(
            [col for col in validation_tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "label"]]
        )
        # Add expert_label column
        validation_tokenized_dataset = validation_tokenized_dataset.add_column("expert_label", [expert_idx] * len(validation_tokenized_dataset))
        print("Validation tokenized dataset before slicing: ", validation_tokenized_dataset['input_ids'].shape, 
              validation_tokenized_dataset['attention_mask'].shape, 
              validation_tokenized_dataset['label'].shape,
              validation_tokenized_dataset['expert_label'].shape)

        if datamix_type == "balanced":
            mixed_validation_dataset_dict["input_ids"] = torch.cat((mixed_validation_dataset_dict["input_ids"], 
                                                                    validation_tokenized_dataset["input_ids"][:take_val]), dim=0)
            mixed_validation_dataset_dict["attention_mask"] = torch.cat((mixed_validation_dataset_dict["attention_mask"], 
                                                                        validation_tokenized_dataset["attention_mask"][:take_val]), dim=0)
            mixed_validation_dataset_dict["label"] = torch.cat((mixed_validation_dataset_dict["label"], 
                                                                validation_tokenized_dataset["label"][:take_val]), dim=0)
            mixed_validation_dataset_dict["expert_label"] = torch.cat((mixed_validation_dataset_dict["expert_label"],
                                                                    validation_tokenized_dataset["expert_label"][:take_val]), dim=0)
        elif datamix_type == "unbalanced":
            mixed_validation_dataset_dict["input_ids"] = torch.cat((mixed_validation_dataset_dict["input_ids"], 
                                                                    validation_tokenized_dataset["input_ids"]), dim=0)
            mixed_validation_dataset_dict["attention_mask"] = torch.cat((mixed_validation_dataset_dict["attention_mask"], 
                                                                        validation_tokenized_dataset["attention_mask"]), dim=0)
            mixed_validation_dataset_dict["label"] = torch.cat((mixed_validation_dataset_dict["label"], 
                                                                validation_tokenized_dataset["label"]), dim=0)
            mixed_validation_dataset_dict["expert_label"] = torch.cat((mixed_validation_dataset_dict["expert_label"],
                                                                    validation_tokenized_dataset["expert_label"]), dim=0)
        else:
            raise ValueError("Invalid datamix type. Please use 'balanced' or 'unbalanced' for datamix type.")
        expert_idx += 1

    return mixed_train_dataset_dict, mixed_validation_dataset_dict

