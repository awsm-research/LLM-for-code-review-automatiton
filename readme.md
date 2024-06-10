# Supplementary material

This supplementary material contains the artifacts of the paper **Fine-Tuning and Prompt Engineering for Large Language Models-based Code Review Automation**.

## Overview

This supplementary material contains the following files and directories:

 -  `script`: this directory contains script that we used in our study.  
    - `run-GPT-3.5-0-shot.py`: for doing 0-shot learning with GPT-3.5
    - `run-GPT-3.5-few-shot.py`: for doing few-shot learning with GPT-3.5
    - `run-fine-tuned-GPT-3.5.py`: for running fine-tuned GPT-3.5
    - `run-Magicoder-0-shot.py`: for doing 0-shot learning with Magicoder
    - `run-Magicoder-few-shot.py`: for doing few-shot learning with Magicoder
    - `fine-tune-Magicoder.py`: for fine-tuning Magicoder
    - `run-fine-tuned-Magicoder.py`: for running fine-tuned Magicoder
    - `create-dataset-for-OpenAI-fine-tuning.py`: for creating dataset to fine-tune GPT-3.5.
    - `evaluation`: this directory contains script for calculating Exact Match and CodeBLEU.

 - `requirement.yml`: a file that stores required python libraries in a conda environment.


The dataset, raw results, and other related files can be obtained from ... . 

## Required libraries

To run our python script, please install the required libraries by using a `conda`, a python environment manager. To do so, please run the following command:

    conda env create -f requirement.yml

## Dataset details

The details of each dataset are as folllows

### dataset-for-few-shot-learning

The dataset is stored as csv file, containing the following columns:

- `input`: the input of a testing set
- `output`: the output of a testing set
- `sample_input_1`, `sample_input_2`, `sample_input_3`: the input examples obtained from a training set
- `sample_output_1`, `sample_output_2`, `sample_output_3`: the output examples obtained from a training set

In the dataset, `sample_input_1`, `sample_input_2`, and `sample_input_3` are paired with `sample_output_1`, `sample_output_2`, and `sample_output_3`, respectively.


### dataset-for-fine-tuning

The dataset is stored as jsonl file. Each record contains the following information:

- persona prompt, which is denoted as *"role": "system"*
- input, which is denoted as *"role": "user"*
- output, which is denoted as *"role": "assistant"*

## Obtaining evaluation results

In this section we explain how to use our script to obtain Exact Match and CodeBLEU

### Obtaining Exact Match

To get the Exact Match results, you can follow the steps below.

 - go to directory `script/evaluation/eval_Exact_Match`
 - run the following command:

    <br/>

    for the CodeReviewer and TufanoT5 dataset
      
		python evaluation.py <dataset_idx> <task_idx> <result_path>

    for the D-ACT dataset
	      
		python evaluation.py <dataset_idx> <task_idx> <proj_idx> <result_path>

The details of the arguments are as follows:


|dataset_idx| dataset |
|--|--|
| 0 | CodeReviewer |
| 1 | TufanoT5 |
| 2 | D-ACT |


<br />

| task_idx | task |
|--|--|
| 0 | Code_Refinement |
| 1 | Code_Refinement_no_comment |



<br />

|proj_idx| project |
|--|--|
| 0 | android |
| 1 | google |
| 2 | ovirt |

Here, `<result_path>` is the path of the generated results

### Obtaining CodeBLEU

1. go to directory `script/evaluation/`

2. run the following command

    <br/>

    for CodeReviewer dataset
    
        python eval_CodeBLEU-multi-lang.py <result_path>

    for TufanoT5 dataset
    
        python eval_CodeBLEU-java.py <dataset_idx> <task_idx> <result_path>
    
    for D-ACT dataset
    
        python eval_CodeBLEU-java.py <dataset_idx> <task_idx> <result_path> <proj_idx>


The details of arguments are as follows:

|dataset_idx| dataset |
|--|--|
| 0 | TufanoT5 |
| 1 | D-ACT |

<br />

| task_idx | task |
|--|--|
| 0 | Code_Refinement |
| 1 | Code_Refinement_no_comment |


<br />

|proj_idx| project |
|--|--|
| 0 | android |
| 1 | google |
| 2 | ovirt |


Here, `<result_path>` is the path of the generated results in the `results` directory.


## Using scripts to run GPT-3.5 model.

In this section we explain how to use our script to do 0-shot learning and few-shot learning , and run fine-tuned GPT-3.5 model


### Using script for zero-shot learning

To do 0-shot learning on CodeReviewer and TufanoT5 dataset, please run the following command:

    python run-GPT-3.5-0-shot.py -d <dataset> -t <task_name> -p_no <prompt_num>

To do 0-shot learning on D-ACT dataset, please run the following command:

    python run-GPT-3.5-0-shot.py -d D-ACT -t CR_NC -p <proj> -p_no <prompt_num>


The argument details are as follows:

- `<dataset>`: CodeReviewer, TufanoT5, D-ACT
- `<task_name>`: CR (Code Refinement (with comment)), CR_NC (Code Refinement (without comment))
- `<proj>`: android, google, ovirt
- `-up`: this argumment indicates that persona will be used.
- `-p_no`: this argument specifies which prompt template will be used for zero-shot learning.


The details of `-p_no` are as follow.

|p_no| meaning |
|--|--|
| 1 | A prompt with simple instruction |
| 2 | A prompt that an instruction is broken into smaller steps |
| 3 | A prompt with detailed instruction |

### Using script for few-shot learning

To do few-shot learning on CodeReviewer and TufanoT5 dataset, please run the following command:

    python run-GPT-3.5-few-shot.py -d <dataset> -t <task_name> -p_no <prompt_num>

To do few-shot learning on D-ACT dataset, please run the following command:

    python run-GPT-3.5-few-shot.py -d D-ACT -t CR_NC -p <proj> -p_no <prompt_num>


The argument details are the same as running 0-shot learning.



### Using script for fine-tuning

To run fine-tuned GPT-3.5 on the CodeReviewer and TufanoT5 dataset, please run the following command:

    python run-fine-tuned-GPT-3.5.py -d <dataset> -t <task_name> -m <model_name>

To run fine-tuned GPT-3.5 on the D-ACT dataset, please run the following command:

    python run-fine-tuned-GPT-3.5.py -d D-ACT -t CR_NC -p <proj> -m <model_name>


Here `<model_name>` is the name of the fine-tuned GPT-3.5 model obtained from OpanAI. 

Since we cannot share the fine-tuned GPT-3.5 model due to OpenAI policy, please fine-tune GPT-3.5 by using the dataset in `dataset-for-fine-tuning` and use our script to run the fine-tuned models.


## Using scripts to run Magicoder model.

In this section we explain how to use our script to do 0-shot learning and few-shot learning, fine-tune Magicoder and run fine-tuned Magicoder model


### Using script for 0-shot learning

To do 0-shot learning on the CodeReviewer and TufanoT5 dataset, please run the following command:

    python run-Magicoder-0-shot.py -d <dataset> -t <task_name>

To do 0-shot learning on the D-ACT dataset, please run the following command:

    python run-Magicoder-0-shot.py -d D-ACT -t CR_NC -p <proj>


The argument details are as follows:

- `<dataset>`: CodeReviewer, TufanoT5, D-ACT
- `<task_name>`: CR (Code Refinement (with comment)), CR_NC (Code Refinement (without comment))
- `<proj>`: android, google, ovirt
- `-up`: this argumment indicates that persona will be used.



### Using script for few-shot learning

To do few-shot learning on the CodeReviewer and TufanoT5 dataset, please run the following command:

    python Magicoder-few-shot.py -d <dataset> -t <task_name>

To do few-shot learning on the D-ACT dataset, please run the following command:

    python Magicoder-few-shot.py -d D-ACT -t CR_NC -p <proj>


The argument details are the same as running 0-shot learning.


### Using script for fine-tuning

To fine-tune Magicoder on the CodeReviewer and TufanoT5 dataset, please run the following command:

    python fine-tune-Magicoder.py -t <task_name> -d <dataset>


To fine-tune Magicoder on the D-ACT dataset, please run the following command:

    python fine-tune-Magicoder.py -t CR_NC -d <dataset> -p <proj>
    
The argument details are the same as running 0-shot learning.

### Using fine-tuned Magicoder models

To run fine-tuned Magicoder models on the CodeReviewer and TufanoT5 dataset, please run the following command:

    python run-fine-tuned-Magicoder.py -d <dataset> -t <task_name> --ckpt_dir <ckpt_dir>

To run fine-tuned Magicoder models on the D-ACT dataset, please run the following command:

    python run-fine-tuned-Magicoder.py -d D-ACT -t CR_NC -p <proj> --ckpt_dir <ckpt_dir>


Here `<ckpt_dir>` is the directory of the model checkpoint that will be used to run Magicoder.

The other argument details are the same as running 0-shot learning.
