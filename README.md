# Welcome to TTLoRAMoE Documentation

This repository correponds to the article  **"TT-LoRA MoE: Unifying Parameter-Efficient Fine-Tuning and Sparse Mixture-of-Experts."** The Contribution of this paper is the introduction of methods to solve inter-task interference and catastrophic forgetting, which commonly arise when fine-tuning large language models (LLMs) for multi-task learning scenarios. To this end, we propose a scalable and reproducible Mixture of Experts (MoE) architecture enhanced with TT-LoRA (Tensor Train Low-Rank Adaptation) and a learned routing mechanism that selects experts dynamically per input. This modular framework ensures parameter efficiency, task specialization, and adaptability to diverse datasets. Below are the contributions of the paper:


[Paper Reference:] (https://arxiv.org/pdf/2504.21190)  


**Contribution 1.** We propose a tensor contraction-based TT-LoRA adaptation that computes Δ𝑊 directly, avoiding full weight reconstruction and improving inference efficiency.  

**Contribution 2.** Our tensor contraction based TT-LoRA PEFT method uses only 2% of LoRA’s parameters and 0.3% of Pfeiffer adapters, while achieving comparable performance, demonstrating strong parameter efficiency.  

**Contribution 3.** We introduce a highly accurate and scalable lightweight router that uses base model representations to produce top-1 logits, mimicking one-hot selection. It outperforms standard routers and scales to 6 experts without loss in performance.  

**Contribution 4.** Our MoE framework preserves the performance of task specific experts and scales to 17 experts when tested individually—unlike AdapterFusion, which suffers from performance drop due to representation interference. 
 
 **Contribution 5.** In multi-task evaluations on mixed datasets, our MoE out performs AdapterFusion, using only 0.03% of Fusion parameters demonstrating greater effectiveness and suitability for handling diverse tasks simultaneously.

### Individual Expert Training
<img src="Expert_training.jpg" alt="Single Training" width="400"/>

### Combined Expert MoE Training
<img src="MoE.jpg" alt="MoE Training" width="800"/>

### Step 1: 
First create a virtual environment in the artifact folder that was cloned using git and follow the codes below:

`conda create --name ttloramoe python=3.11` this creates a virtual environment as ttloramoe

`conda activate ttloramoe` this activates the ttloramoe environment

`pip install pytorch-lightning pandas tensorly datasets transformers` this installs the required packages

`pip install transformers[torch] accelerate>=0.26.0`

### Step 2:
`python download_model.py` this downloads the llama3.2-1b model checkpoints and saves to your current local directory

#### Note:
All the required datasets are provided inside the datasets folder. There are 17 supported datasets. Moreover, the supported command line options for `--dataset` to run any of the below listed experiments (where
applicable) is:
`--dataset` [17 options]:
`mrpc, cola, sst2, rte, cb, sick, boolq, csqa, winogrande_l, scitail, imdb, cosmosqa, socialiqa, hellaswag, qnli, qqp, mnli`

Before running any experiments that are described below, the following setup should be completed step by step:
1. Allocate the GPU
2. Create the virtual environment and install required packages as mentioned above
3. Download the model as mentioned above
4. It’s ready to execute the experiments

### Experiment 1 to support Contribution 1: 
#### Inference Time Comparison between Conventional Reconstruction and Proposed Contraction Method
Inside the folder Artifact_1.1, the script in `inference_comparison.py` contains two TT-LoRA adaptation of loaded pre-trained model: conventional reconstruction and proposed contraction method.
Running terminal inside the cloned folder from git, we can execute the code in the given format:
`python Artifact_1.1/inference_comparison.py --batchsize 8 --dataset qnli --test reconstruction --gpus 4 --workers 8`  

reproducing results as in Table 3 from the paper, cli options are:
'# for `--batchsize [2,4,6,8,16,32,64,128]`
'# for `--test [reconstruction, contraction]`
'# for `--dataset` options are provided above 
'# we used `--gpus 4` and `--workers 8` as default settings

we used `qnli` as `--dataset` option because to process 10 iterations of 128 samples, some datasets might not have enough samples. We can also bash run the experiment using the command line as:
`bash inference_comparison_bash.sh` to get all the comparisons between the two types of TT-LoRA adaptation altogether for all the batch sizes.

### Experiment 2 to support Contribution 2: 
#### Training Individual Experts using TTLoRA, LoRA and Adapters
Inside the folder Artifact_1.2, there are scripts to fine tune base model using 3 different PEFT methods - LoRA, TT-LoRA and Adapter. The results from the paper provided in Table 4 and Table 6 (under the Single Training head) can be generated using this artifact.  
##### LoRA Training on Single Dataset: 
The script `single_LoRA_training.py` fine tunes the base model Llama3.2-1b using LoRA adaptation with ray tuned hyper-parameters 𝑟𝑎𝑛𝑘 = 16, 𝑎𝑙𝑝ℎ𝑎 = 8 and 𝑙𝑒𝑎𝑟𝑛𝑖𝑛𝑔_𝑟𝑎𝑡𝑒 = 0.0005. We can run the
script as `python Artifact_1.2/single_LoRA_training.py --gpus 4 --workers 8 --epochs 100 --patience 10 --batchsize 32 --dataset mrpc`  
'# for --dataset , options are provided above in the note 

##### TTLoRA Training on Single Dataset: 
The script `single_TTLoRA_training.py` fine tunes the base model Llama3.2-1b using TTLoRA adaptation with ray tuned hyper-parameters 𝑟𝑎𝑛𝑘 = 5, 𝑎𝑙𝑝ℎ𝑎 = 16, 𝑄 𝑠ℎ𝑎𝑝𝑒 = [64, 4, 3, 3, 4, 64], 𝑉 𝑠ℎ𝑎𝑝𝑒 =
[64, 4, 3, 3, 4, 64] and 𝑙𝑒𝑎𝑟𝑛𝑖𝑛𝑔_𝑟𝑎𝑡𝑒 = 0.005. We can run the script as `python Artifact_1.2/single_TTLoRA_training.py --gpus 4 --workers 8 --epochs 100 --patience 10 --batchsize 32 --dataset mrpc`  
'# for --dataset , options are provided above in the note 

##### Adapter Training on Single Dataset: 
The script `single_Adapter_training.py` fine tunes the base model Llama3.2-1b using Adapter adaptation with provided hyper-parameter 𝑙𝑒𝑎𝑟𝑛𝑖𝑛𝑔_𝑟𝑎𝑡𝑒 = 0.0005 in their work. We can run the script as `python Artifact_1.2/single_Adapter_training.py --gpus 4 --workers 8 --epochs 100 --patience 10 --batchsize 32 --dataset mrpc`  
'# for --dataset , options are provided above in the note 

### Experiment 3 to support Contribution 3: 
#### Comaparsion of different types of router
Inside the folder Artifact_1.3 the script `moe_train.py` lets us test the effectiveness of 3 different router types - single layer, multi-layer and our proposed (single trainable layer augmented with base model’s representation) as mentioned in contribution 3 and as presented the results in Table 5 in the paper.

Before running this artifact, we need to make sure we have the TT-LoRA trained experts saved in `TTLoRA_Saved_Individual_Expert` under the folder names as that of dataset names. We assume there are at least these saved checkpoints for `MNLI, CoLA, RTE, SST2, QNLI, and QQP` to reproduce our results presented in Table 5. Before running the script, we need to fix some of the variables in the main
function. Inside `main()` function (don’t change in config), there are variables: `dataload_type`, `multiple_datasets`, and `experts_list`. First, we fix the `experts_list` by providing the list of experts which
are trained and present in the `TTLoRA_Saved_Individual_Expert` folder. The router makes the decision among these experts. Then, we fix the `dataload_type` by providing `multiple` as the option as
this loads the mixture of the dataset mentioned in `multiple_datasets` list. If the `dataload_type` is `single` then it only loads the dataset which we provide through cli command using `--dataset`. And we change the `multiple_datasets` list with the datasets to be combined of the experts present in the `experts_list`. 

We can also try different combinations of experts and different combinations of multiple datasets or single dataset, we just need to make sure that we have trained experts if we use that in `experts_list`. We can now run the moe_train.py script using the following cli command:   
`python Artifact_1.3/moe_train.py --batchsize 32 --epochs 100 --patience 10 --workers 8 --gpus 4 --router llm`  
'# --router [options: `llm`, `single_layer`, `multi_layer`]  
'# we don't use --dataset option if "multiple" option is chosen as "dataload_type"

### Experiment 4 to support Contribution 4: 
#### Training MoE in Single Dataset

The Artifact_1.3 folder contains the `moe_train.py` file which lets us train and test the performance of our proposed TTLoRA-MoE, when loaded with individually pre-trained experts, on single dataset and see if the MoE can retain the individual performances of the pretrained experts as mentioned in contribution 4. Similarly, the folder also contains the script `adapterfusion.py` which lets us train and test the fusion layer’s performance in the AdapterFusion to see if it can retain the performance of individually pre-trained adapters. The results as presented in the Table 6 under Expert Integration heading can be taken as reference.
##### Running TTLoRA-MoE on Single Dataset:
Before running the script moe_train.py, we need to make sure that we have required experts already trained and stored in `TT-LoRA_Saved_Individual_Expert` folder. Moreover the script only expects one checkpoint for each expert, so we need to make sure there’s only one checkpoint for each trained expert. Now, in the `moe_train.py` script, we need to change some variables in the `main()` function. First, we change the `dataload_type` into `single` and then we make sure the `experts_list` variable consists the list of the pre-trained experts that we want to put together (the ones which are already trained and reside inside the `TTLoRA_Saved_Individual_Expert` folder). Then we can run the code using the cli as:  
`python Artifact_1.3/moe_train.py --batchsize 16 --epochs 100 --patience 10 --workers 8 --gpus 4 --dataset mrpc --router llm`  
'# for --dataset , options are provided above in the note 

##### Running AdapterFusion on Single Dataset:
Before running the script adapterfusion.py, we need to make sure that we have required adapters already trained and stored in `Adapters_Saved_Individual_Expert` folder inside respective folder as `{datasetname}_seq_bn`. We ignore the folder `Adapter_Checkpoints` folder as it has no any part in AdapterFusion. Each saved adapter’s folder consists of the saved trained configuration of each adapter. Moreover the script only expects one configuration for each expert, so we need to make sure there’s only one configuration for each trained expert. Now, in the `adapterfusion.py` script, we need to change some variables in the `train_moe_without_ray(config)`
function and `main()` function.  

First, in `train_moe_without_ray` function, we need to make sure the `model.load_adapter` function includes only the trained adapters that are present in the `Adapter_Saved_Individual_Expert` folder. Then we need to define the `fusion_of` variable with the adapters that we want to fuse together for the experiments. And we use the same list of adapters in `adapter_setup` variable as well. Then based on available GPU and time, we need to fix the `num_train_epochs` variable as 10 (as authors mentioned they performed 10 epochs for every fusion test) or less. Along with this we also need to fix `per_device_train_batch_size` and `per_device_eval_batch_size` 1 to
32 based on the dataset and available GPU. Some of the datasets like `imdb, boolq, mnli, qqp, qnli` take very less number of batchsizes and also take substantial time to run 10 epochs. So, for testing purpose, we suggest to perform less number of epochs if these datasets are to be tested in the Fusion environment. Though this issue is not seen in our proposed TTLoRA-MoE setting.  

Then, in `main()` function, we change the `dataload_type` into `single` and pass the dataset name through cli as:
`python Artifact_1.3/adapterfusion.py --workers 8 --gpus 4 --dataset mrpc`  
'# for --dataset , options are provided above in the note 

### Experiment 5 to support Contribution 5: 
#### Training MoE in Mixed Dataset
##### Running TTLoRA MoE on Mixed Dataset:
All the process are same as above as mentioned in Experiment 4, in here we just change the `dataload_type` into `multiple` as the option and this loads the mixture of the dataset mentioned in `multiple_datasets` list. And we change the `multiple_datasets` list with the datasets to be combined of the experts present in the `experts_list`. We can also try different combinations of experts and different combinations of `multiple datasets`.
Then we make sure the experts_list variable consists the list of the pre-trained experts that we want to put together (the ones which are already trained and reside inside the `TTLoRA_Saved_Individual_Expert` folder). Then we can run the code using the cli as:  
`python Artifact_1.3/moe_train.py --batchsize 16 --epochs 100 --patience 10 --workers 8 --gpus 4 --router llm`  
'# for --dataset , options are provided above in the note

##### Running AdapterFusion on Mixed Dataset:
Similar to experiment 4, in here we just change the `dataload_type` into `multiple` and run the script as `python Artifact_1.3/adapterfusion.py --workers 8 --gpus 4`  
'# for --dataset , options are provided above in the note



## Authors

* [Pradip Kunwar] (mailto:akabir4@gmu.edu)- Theoretical Division, Los Alamos National Laboratory
* [Manish Bhattarai] (mailto:ceodspspectrum@lanl.gov)- Theoretical Division, Los Alamos National Laboratory

## How to cite TTLoRAMoE?
```latex
@article{kunwar2025tt,
  title={TT-LoRA MoE: Unifying Parameter-Efficient Fine-Tuning and Sparse Mixture-of-Experts},
  author={Kunwar, Pradip and Vu, Minh N and Gupta, Maanak and Abdelsalam, Mahmoud and Bhattarai, Manish},
  journal={arXiv preprint arXiv:2504.21190},
  year={2025}
}

```

## Acknowledgements

Los Alamos National Lab (LANL), T-1

## Copyright notice

© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

LANL O#4926

## License

This program is Open-Source under the BSD-3 License.
 
* Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


