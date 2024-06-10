#%%

import pandas as pd

from tqdm import tqdm

from peft import PeftModel

import torch
import transformers
import argparse, os, sys, json, logging, shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model, 
    LoraConfig
)


#%%

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--dataset', '-d', type=str, required=True)
parser.add_argument('--proj', '-p', type=str, default = 'android')
parser.add_argument('--task_name', '-t', type=str, required=True)
parser.add_argument('--ckpt_num', type=int, required=True)
parser.add_argument('--use_persona', '-up', action='store_true')

main_data_path = '../dataset-rearranged/'


data_path = {
    'CodeReviewer-paper': {
        'Code_Refinement': os.path.join(main_data_path, 'CodeReviewer-paper/Code_Refinement/test/CodeReviewer-test-data-for-OpenAI-finetuning.jsonl')
    },
    'TufanoT5-paper': {
        'Code_Refinement': os.path.join(main_data_path, 'TufanoT5-paper/Code_Refinement/test/TufanoT5-with-comment-test-data-for-OpenAI-finetuning.jsonl'),
        'Code_Refinement_no_comment': os.path.join(main_data_path, 'TufanoT5-paper/Code_Refinement_no_comment/test/TufanoT5-test-data-without-comment-for-OpenAI-finetuning.jsonl')
    },
    'D-ACT-paper': {
        'Code_Refinement_no_comment': os.path.join(main_data_path, 'D-ACT-paper/Code_Refinement_no_comment/dataset-time-wise/{}/test/D-ACT-{}-test-data-for-OpenAI-finetuning-small-without-persona.jsonl')
    }
}


dataset_names = ['CodeReviewer-paper', 'TufanoT5-paper', 'D-ACT-paper']

modes = ['Code_Refinement', 'Code_Refinement_no_comment']

projs = ['android', 'google', 'ovirt']

task_name_map = {
    'CR': 'Code_Refinement',
    'CR_NC': 'Code_Refinement_no_comment'
}

args = parser.parse_args()


dataset_name = args.dataset
proj = args.proj
task_name = args.task_name
ckpt_num = args.ckpt_num
is_use_persona = args.use_persona

mode = task_name_map[task_name]

output_dir = '../output/{}/{}/'.format(dataset_name, task_name)

os.makedirs(output_dir, exist_ok = True)

backup_dir = os.path.join(output_dir,'backup')


backup_dir = os.path.join(output_dir,'backup')


suffix = ''

if is_use_persona:
    suffix = 'with-persona'
    print('run model with persona')
else:
    suffix = 'without-persona'
    print('run model without persona')


if dataset_name == 'D-ACT-paper':
    
    file_path = data_path[dataset_name][mode].format(proj, proj)
    model_dir = "../model/D-ACT-paper/Code_Refinement_no_comment/dataset-time-wise/{}/magicoder-ckpt-new-{}/".format(proj, suffix)
    output_dir = '../output/D-ACT-paper/Code_Refinement_no_comment/dataset-time-wise/{}/'.format(proj)


else:
    file_path = data_path[dataset_name][mode]
    model_dir = '../model/{}/{}/magicoder-ckpt-new-{}'.format(dataset_name, mode, suffix)
    output_dir = '../output/{}/{}/'.format(dataset_name, mode)

output_file_path = os.path.join(output_dir, 'output_from_magicoder_fine-tuned_{}.csv'.format(suffix))

# all_results = all_results[:round(len(all_results)/2)]
# em_old_prompt = em_old_prompt[:round(len(em_old_prompt)/2)]

backup_file_path = os.path.join(backup_dir, 'output_from_magicoder_fine-tuned_{}_backup.csv'.format(suffix))

os.makedirs(backup_dir, exist_ok=True)

with open(file_path) as f:
    data_lines = f.readlines()



if os.path.exists(output_file_path):
    print('already have', output_file_path)
    output_df = pd.read_csv(output_file_path)
    output_df = output_df.fillna('<empty>')

    print('already finished {}/{}'.format(len(output_df[output_df['output']!='<empty>']), len(output_df)))

else:
    print('first time to have', output_file_path)
    output_df = pd.DataFrame()
    output_df['input'] = ['<empty>']*len(data_lines)
    output_df['output'] = '<empty>'


# %%


model_name = "ise-uiuc/Magicoder-S-DS-6.7B"

model = AutoModelForCausalLM.from_pretrained(model_name,
    # load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if is_use_persona:

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

    print('change template for using my persona')
    print('the new template is')
    print(template)

    tokenizer.chat_template = template

peft_model_id = model_dir+'/checkpoint-{}'.format(ckpt_num)

print('load DoRA adapter from', peft_model_id)

peft_model = PeftModel.from_pretrained(model, peft_model_id, offload_folder="lora_results/lora_7/temp")

print('load model finished')

#%%

all_inputs = output_df['input'].tolist()
all_results = output_df['output'].tolist()


#%%

for idx, row in tqdm(enumerate(data_lines)):
    
    if all_results[idx] == '<empty>':

        if os.path.exists(output_file_path):
            shutil.copy(output_file_path, backup_file_path)

        json_data = json.loads(row)

        system_prompt = json_data['messages'][0]['content']
        model_input = json_data['messages'][1]['content']

        all_model_inputs = []

        if is_use_persona:
            all_model_inputs.append({"role": "system", "content": system_prompt})

        all_model_inputs.append({"role": "user", "content": model_input})

        actual_model_input = tokenizer.apply_chat_template(all_model_inputs, tokenize=False)

        try:

            tokenized_input = tokenizer(actual_model_input, return_tensors="pt")
            
            generation_output = peft_model.generate(
                input_ids=tokenized_input["input_ids"].to("cuda"),
                attention_mask = tokenized_input['attention_mask'].to("cuda"),
                max_new_tokens=256,
                do_sample=True,
                top_k=0,
                top_p=1,
                temperature=0.00000001,
                repetition_penalty=0.00000001,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)

            # print('model input')
            # print(tokenizer.apply_chat_template(all_model_inputs, tokenize=False))
            # print(actual_model_input)
            # print('-'*30)
            # print('generated text')
            # print(generated_text)
            # print('='*40)
        
        except Exception as e:
            logging.error("Running {}\nAn error occurred: {}".format(output_file_path, str(e)))
            print('error')
            print(e)
            print('-'*30)
            generated_text = '<empty>'
            # pass

        if is_use_persona:
            all_inputs[idx] = system_prompt + '\n\n' + model_input
        else:
            all_inputs[idx] =  model_input

        
        all_results[idx] = generated_text

        output_df['input'] = all_inputs
        output_df['output'] = all_results

        output_df.to_csv(output_file_path, index=False)

#%%

output_df.to_csv(output_file_path, index=False)