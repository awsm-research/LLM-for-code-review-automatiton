#%%

import os, argparse, shutil, httpx
import pandas as pd
from tqdm import tqdm

import torch
from transformers import pipeline

#%%

task_name_map = {
    'CR': 'Code_Refinement',
    'CR_NC': 'Code_Refinement_no_comment'
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
num_sample = 3

suffix = ''

if is_use_persona:
    suffix = 'with-persona'
    print('run model with persona')
else:
    suffix = 'without-persona'
    print('run model without persona')

#%%
main_data_dir = '../datasets/dataset-for-few-shot-learning/'

data_paths = {
    'CodeReviewer-paper': {
        'with-comment': os.path.join(main_data_dir,'CodeReviewer-paper/test_df_with_samples.csv')
    },
    'TufanoT5-paper': {
        'with-comment': os.path.join(main_data_dir,'TufanoT5-paper/with-comment/test_df_with_samples.csv'),
        'without-comment': os.path.join(main_data_dir,'TufanoT5-paper/without-comment/test_df_with_samples.csv')
    },
    'D-ACT-paper': {
        'without-comment': os.path.join(main_data_dir,'D-ACT-paper/{}/test_df_with_samples.csv')
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

output_file_path = os.path.join(output_dir, 'few-shot-{}.csv'.format(suffix))
backup_file_path = os.path.join(backup_dir, 'few-shot-{}.csv'.format(suffix))

test_set_df = pd.read_csv(test_set_path)

## resume mechanism
if os.path.exists(output_file_path):
    print('already have', output_file_path)
    output_df = pd.read_csv(output_file_path)
    output_df = output_df.fillna('<empty>')

    print('already finished {}/{}'.format(len(output_df[output_df['output']!='<empty>']), len(output_df)))

else:
    print('first time to have', output_file_path)

    output_df = pd.DataFrame()
    output_df['input'] = test_set_df['input'].to_list()
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



persona_prompt = {
    'multi-lang': 'You are an expert software developer in {}. You always want to improve your code to have higher quality.',
    'java': 'You are an expert software developer in java. You always want to improve your code to have higher quality.'
}

prompt_header_with_comment = {
    'multi-lang': '''You are given 3 examples. Each example begins with "##Example" and ends with "---". Each example contains the submitted code, the developer comment, and the improved code. The submitted code and improved code is written in {}. Your task is to improve your submitted code based on the comment that another developer gave you.\n\n''',
    'java': '''You are given 3 examples. Each example begins with "##Example" and ends with "---". Each example contains the submitted code, the developer comment, and the improved code. The submitted code and improved code is written in Java. Your task is to improve your submitted code based on the comment that another developer gave you.\n\n'''
}


prompt_header_without_comment = '''You are given 3 examples. Each example begins with "##Example" and ends with "---". Each example contains the submitted code and the improved code. The submitted code and improved code is written in Java. Your task is to improve your submitted code.\n\n'''

example_prompt_by_task = {
    'Code_Refinement': '''## Example\n\nSubmitted code:"""{}"""\n\nDeveloper comment: """{}"""\n\nImproved code: """{}"""\n\n---\n\n''',
    'Code_Refinement_no_comment': '''## Example\n\nSubmitted code:"""{}"""\n\nImproved code: """{}"""\n\n---\n\n''',
}

main_prompt_by_task = {
     'Code_Refinement': '''Submitted code:"""{}"""\n\nDeveloper comment: """{}"""\n\nImproved code: """''',
    'Code_Refinement_no_comment': '''Submitted code:"""{}"""\n\nImproved code: """'''
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

def get_persona_prompt(lang):
    ret = ''
    if dataset_name == 'CodeReviewer-paper':
        ret = persona_prompt['multi-lang'].format(lang)
    else:
        ret = persona_prompt['java']

    return ret

def get_prompt_header(lang):

    ret = ''

    if task_name == 'Code_Refinement':
        if dataset_name == 'CodeReviewer-paper':
            ret = prompt_header_with_comment['multi-lang'].format(lang)
        else:
            ret = prompt_header_with_comment['java']
    else:
        ret = prompt_header_without_comment

    return ret


def get_sample_str(row):

    ret = ''

    for i in range(1,num_sample+1):

        if task_name == 'Code_Refinement':
            if dataset_name == 'CodeReviewer-paper':
                code, comment = row['sample_input_{}'.format(i)].split('<SEP_DATA>')
            else:
                code, comment = row['sample_input_{}'.format(i)].split('<SEP>')

            sample_prompt = example_prompt_by_task[task_name].format(code, comment, row['sample_output_{}'.format(i)])

        else:
            sample_prompt = example_prompt_by_task[task_name].format(row['sample_input_{}'.format(i)], row['sample_output_{}'.format(i)])

        ret = ret + sample_prompt

    return ret


def get_main_prompt(row):

    ret = ''

    if task_name == 'Code_Refinement':
        if dataset_name == 'CodeReviewer-paper':
            code, comment = row['input'].split('<SEP_DATA>')
        else:
            code, comment = row['input'].split('<SEP>')
            
        ret = main_prompt_by_task['Code_Refinement'].format(code, comment)
    else:
        code = row['input']
        ret = main_prompt_by_task['Code_Refinement_no_comment'].format(code)

    return ret


#%%

generator = pipeline(
    model="ise-uiuc/Magicoder-S-DS-6.7B",
    task="text-generation",
    torch_dtype=torch.float16,
    device_map="auto",
)

#%%

for idx, row in tqdm(test_set_df.iterrows()):

    if all_results[idx] == '<empty>':

        if os.path.exists(output_file_path):
            shutil.copy(output_file_path, backup_file_path)

        if dataset_name == 'CodeReviewer-paper':
            lang = lang_dict[lang_list[idx]]
        else:
            lang = 'java'

        ## create backup file in case program is interrupt and file is broken
        if os.path.exists(output_file_path):
            shutil.copy(output_file_path, backup_file_path)

        prompt = '@@ Instruction\n\n'

        if is_use_persona:
            real_persona_prompt = get_persona_prompt(lang)

            prompt = real_persona_prompt + '\n\n' + prompt

        prompt = prompt + get_prompt_header(lang) + get_sample_str(row) + get_main_prompt(row)
        
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
