#%%

import torch
import transformers
import argparse, os, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model, 
    LoraConfig
)

from trl import SFTTrainer


#%%

main_data_path = '../dataset-for-fine-tuning/'


data_path = {
    'CodeReviewer-paper': {
        'Code_Refinement': {
            'train': os.path.join(main_data_path, 'CodeReviewer-paper/train/dataset-{}.jsonl'), 
            'valid': os.path.join(main_data_path, 'CodeReviewer-paper/valid/dataset-{}.jsonl')
        }
    },
    'TufanoT5-paper': {
        'Code_Refinement': {
            'train': os.path.join(main_data_path, 'TufanoT5-paper/with-comment/train/dataset-{}.jsonl'), 
            'valid': os.path.join(main_data_path, 'TufanoT5-paper/with-comment/valid/dataset-{}.jsonl')
        },
        'Code_Refinement_no_comment': {
            'train': os.path.join(main_data_path, 'TufanoT5-paper/without-comment/train/dataset-{}.jsonl'), 
            'valid': os.path.join(main_data_path, 'TufanoT5-paper/without-comment/valid/dataset-{}.jsonl')
        }
    },
    'D-ACT-paper': {
        'Code_Refinement_no_comment': {
            'train': os.path.join(main_data_path, 'D-ACT-paper/{}/train/dataset-{}.jsonl'), 
            'valid': os.path.join(main_data_path, 'D-ACT-paper/{}/valid/dataset-{}.jsonl')
        }
    }
}

dataset_names = ['CodeReviewer-paper', 'TufanoT5-paper', 'D-ACT-paper']

modes = {
    'CR': 'with-comment',
    'CR_NC': 'without-comment'
}

projs = ['android', 'google', 'ovirt']

# dataset = dataset_names[0]
# mode = modes[0]

parser = argparse.ArgumentParser()

# Add arguments
# parser.add_argument('--model_name', '-m', type=str, required=False, default = 'gpt-3.5-turbo') # gpt-3.5-turbo or gpt-4
parser.add_argument('--task_name', '-t', type=str, required=True) # CR: 'Code_Refinement', CG: 'Comment_Generation', CR_NC: 'Code_Refinement_no_comment'
parser.add_argument('--dataset', '-d', type=str, required=True)
parser.add_argument('--proj', '-p', type=str, default='')
parser.add_argument('--use_persona', '-up', action='store_true')
parser.add_argument('--resume_training', '-res', action='store_true')

args = parser.parse_args()

# dataset = dataset_names[int(sys.argv[1])]
# mode = modes[int(sys.argv[2])]

dataset = args.dataset
mode = modes[args.task_name]
use_persona = args.use_persona
proj = args.proj
resume_training = args.resume_training


suffix = ''

if use_persona:
    suffix = 'with-persona'
else:
    suffix = 'without-persona'



if dataset == 'D-ACT-paper':
    proj = args.proj

    train_file_path = data_path[dataset][mode]['train'].format(proj, suffix)
    valid_file_path = data_path[dataset][mode]['valid'].format(proj, suffix)


else:
    train_file_path = data_path[dataset][mode]['train'].format(suffix)
    valid_file_path = data_path[dataset][mode]['valid'].format(suffix)


print('load train data from', train_file_path)
print('load validation data from', valid_file_path)


#%%

train_dataset = load_dataset("json", data_files=train_file_path)
valid_dataset = load_dataset("json", data_files=valid_file_path)

## data is not split, so everything is in 'train'
train_dataset = train_dataset['train']
valid_dataset = valid_dataset['train']


#%%

model_name = "ise-uiuc/Magicoder-S-DS-6.7B"

model = AutoModelForCausalLM.from_pretrained(model_name,
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if use_persona:

    print('change template for using my persona')

    suffix = 'with-persona'

    template = '''{{bos_token}}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {{message['content'] + '

'}}
    {%- else %}
        {%- if message['role'] == 'user' %}
{{'@@ Instruction
' + message['content'] + '

'}}
        {%- else %}
{{'@@ Response
' + message['content'] + eos_token + '

'}}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{{'@@ Response
'}}'''

    tokenizer.chat_template = template

else:
    suffix = 'without-persona'


print('load model finished')


#%%

lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True
    )


#%%

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print('load model for dora finished')


#%%

# output_dir = ""

if dataset == 'D-ACT-paper':
    output_dir = "../model/D-ACT-paper/Code_Refinement_no_comment/dataset-time-wise/{}/magicoder-ckpt-new-{}/".format(proj, suffix)

else:
    output_dir = '../model/{}/{}/magicoder-ckpt-new-{}'.format(dataset, mode, suffix)



per_device_train_batch_size = 12
gradient_accumulation_steps = 1
per_device_eval_batch_size = 12
eval_accumulation_steps = 1
optim = "paged_adamw_32bit"
save_steps = 100 # default is 100 (same as GPT-3.5)
logging_steps = 100 # default is 100 (same as GPT-3.5)
learning_rate = 5e-4
max_grad_norm = 0.3
max_steps = -1
warmup_ratio = 0.03
evaluation_strategy="steps"
lr_scheduler_type = "constant"

training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        evaluation_strategy=evaluation_strategy,
        save_steps=save_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        ddp_find_unused_parameters=False,
        eval_accumulation_steps=eval_accumulation_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs = 10
    )


#%%

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=lora_config,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args
)


if resume_training:
    print('resume model training from the last checkpoint')

trainer.train(resume_from_checkpoint = resume_training)

trainer.save_model(f"{output_dir}/final")


# %%
