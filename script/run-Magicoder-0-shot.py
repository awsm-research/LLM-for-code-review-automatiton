#%%

import os, argparse, shutil, httpx
import pandas as pd
from tqdm import tqdm

import torch
from transformers import pipeline

#%%

task_name_map = {
    'CR': 'with-comment',
    'CR_NC': 'without-comment'
}

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', '-d', type = str, required=True)
parser.add_argument('--task_name', '-t', type=str, required=True)
parser.add_argument('--proj', '-p', type=str, default='')
parser.add_argument('--use_persona', '-up', action='store_true')

args = parser.parse_args()

dataset_name = args.dataset
task_name = task_name_map[args.task_name]
proj = args.proj
is_use_persona = args.use_persona

max_len = 512

suffix = ''

if is_use_persona:
    suffix = 'with-persona'
    print('run model with persona')
else:
    suffix = 'without-persona'
    print('run model without persona')

#%%

main_data_dir = '../datasets/original-dataset/'

data_paths = {
    'CodeReviewer-paper': {
        'with-comment': os.path.join(main_data_dir,'CodeReviewer-paper/test/input.txt')
    },
    'TufanoT5-paper': {
        'with-comment': os.path.join(main_data_dir,'TufanoT5-paper/with-comment/test/input.txt'),
        'without-comment': os.path.join(main_data_dir,'TufanoT5-paper/without-comment/test/input.txt')
    },
    'D-ACT-paper': {
        'without-comment': os.path.join(main_data_dir,'D-ACT-paper/{}/test/input.txt')
    }
}

if dataset_name in ['CodeReviewer-paper', 'TufanoT5-paper']:
    test_set_path = data_paths[dataset_name][ task_name]
else:
    test_set_path = data_paths[dataset_name][ task_name].format(proj)

output_dir = '../output/{}/{}/'.format(dataset_name, task_name)
backup_dir = '../output/{}/{}/backup/'.format(dataset_name, task_name)

os.makedirs(output_dir, exist_ok = True)
os.makedirs(backup_dir, exist_ok = True)


suffix = ''

if is_use_persona:
    suffix = 'with-persona'
else:
    suffix = 'without-persona'

output_file_path = os.path.join(output_dir, '0-shot-{}.csv'.format(suffix))
backup_file_path = os.path.join(backup_dir, '0-shot-{}.csv'.format(suffix))

## resume mechanism
if os.path.exists(output_file_path):
    print('already have', output_file_path)
    output_df = pd.read_csv(output_file_path)
    output_df = output_df.fillna('<empty>')

    print('already finished {}/{}'.format(len(output_df[output_df['output']!='<empty>']), len(output_df)))

else:
    print('first time to have', output_file_path)

    with open(test_set_path) as f:
        lines = f.readlines()

    if dataset_name == 'CodeReviewer':
        lines = [s.replace('<NEW_LINE>','\n') for s in lines]

    output_df = pd.DataFrame()
    output_df['input'] = lines
    output_df['actual_input_prompt'] = '<empty>'
    output_df['output'] = '<empty>'


all_results = output_df['output'].tolist()
actual_input_prompts = output_df['actual_input_prompt'].tolist()


#%%

if dataset_name == 'CodeReviewer-paper':
    double_slash_comment_lang = ['.cs', 'c', 'cpp', 'go', 'java', 'js', 'php']

    with open('../dataset-rearranged/CodeReviewer-paper/{}/test/lang.txt'.format(task_name)) as f:
        lang_list = f.readlines()

    lang_list = [s.replace('\n','').strip() for s in lang_list]


main_prompt_by_task = {
    'Code_Refinement': 'Your task is to improve the given submitted code based on the given reviewer comment. Please only generate the improved code without your explanation.\n\n"""{}"""',
    'Code_Refinement_no_comment': 'Your task is to improve the given submitted code. Please only generate the improved code without your explanation.\n\n"""{}"""'
}

persona_prompt = {
    'multi-lang': 'You are an expert software developer in {}. You always want to improve your code to have higher quality.',
    'java': 'You are an expert software developer in java. You always want to improve your code to have higher quality.'
}

lang_dict = {
    'py': 'Python',
    'c': 'C',
    'go': 'Go',
    'js': 'Javascript',
    'java': 'Java',
    '.cs': 'C#',
    'php': 'php',
    'cpp': 'C++',
    'rb': 'Ruby'
}


#%%

generator = pipeline(
    model="ise-uiuc/Magicoder-S-DS-6.7B",
    task="text-generation",
    torch_dtype=torch.float16,
    device_map="auto",
)


#%%

for idx, row in tqdm(output_df.iterrows()):

    if all_results[idx] == '<empty>':

        if os.path.exists(output_file_path):
            shutil.copy(output_file_path, backup_file_path)

        if dataset_name == 'CodeReviewer-paper':
            lang = lang_dict[lang_list[idx]]

        ## create backup file in case program is interrupt and file is broken
        if os.path.exists(output_file_path):
            shutil.copy(output_file_path, backup_file_path)

        prompt = '@@ Instruction\n\n'
        

        if is_use_persona:
            if dataset_name == 'CodeReviewer-paper':
                prompt = persona_prompt['multi-lang'].format(lang) + '\n\n' + prompt
            else:
                prompt = persona_prompt['java'].format(lang) + '\n\n' + prompt


        if task_name == 'Code_Refinement':
            if dataset_name == 'CodeReviewer-paper':
                code, comment = row['input'].split('<SEP_DATA>')
                comment_line = '//Reviewer comment: {}'.format(comment) if lang in double_slash_comment_lang else '#Reviewer comment: {}'.format(comment)
                prompt = main_prompt_by_task[task_name].format(code + '\n\n' + comment_line)
            else:
                code, comment = row['input'].split('<SEP>')
                prompt = main_prompt_by_task[task_name].format(code + '\n\n//Reviewer comment: ' + comment)
        else:
            code = row['input']
            prompt = main_prompt_by_task[task_name].format(code)

        prompt = prompt + '\n\n@@ Response \n\n'

        try:
            result = generator(prompt, max_length=2048, num_return_sequences=1, temperature=0.0)

            generated_text = result[0]["generated_text"]
        
        except Exception as e:
            print('error')
            print(e)
            print('-'*30)
            generated_text = '<empty>'

        actual_input_prompts[idx] = prompt
        all_results[idx] = generated_text

        output_df['actual_input_prompt'] = actual_input_prompts
        output_df['output'] = all_results

        output_df.to_csv(output_file_path, index=False)
        

#%%

output_df.to_csv(output_file_path, index=False)
# %%
