#%%

import os, argparse, shutil, httpx
import pandas as pd
from tqdm import tqdm

from openai import OpenAI

API_KEY = "<YOUR API KEY>"

client = OpenAI(api_key = API_KEY, timeout=httpx.Timeout(60.0, read=10.0, write=10.0, connect=5.0))

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
parser.add_argument('--prompt_num', '-p_no', type = int, required = True)

args = parser.parse_args()

dataset_name = args.dataset
task_name = task_name_map[args.task_name]
proj = args.proj
is_use_persona = args.use_persona
prompt_num = args.prompt_num


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



persona_prompt = 'You are an expert software developer in {}. You always want to improve your code to have higher quality.'


all_prompt_header_with_comment = {
    1: '''You are given 3 examples. Each example begins with "##Example" and ends with "---". Each example contains the submitted code, the developer comment, and the improved code. The submitted code and improved code is written in {}. Your task is to improve your submitted code based on the comment that another developer gave you.\n\n''',
    2: '''You are given 3 examples. Each example begins with "##Example" and ends with "---". Each example contains the submitted code, the developer comment, and the improved code. The submitted code and improved code is written in {}.
        
Follow the steps below to improve the given submitted code
step 1 - read the given submitted code and reviewer comment in the above examples
step 2 - identify lines that need to be modified, added or deleted in the examples
step 3 - read the given submitted code and reviewer comment
step 4 - identify lines that need to be modified, added or deleted
step 5 - generate the improved code without your explanation.\n\n''',
    3: '''You are given 3 examples. Each example begins with "##Example" and ends with "---". Each example contains the submitted code, the developer comment, and the improved code. The submitted code and improved code is written in {}.
    
A developer asks you to help him improve his submitted code based on the given reviewer comment. He emphasizes that the improved code must have higher quality, conforms to coding convention or standard, and works correctly. He tells you to refrain from putting the submitted code in a class or method, and providing global variables or an implementation of methods that appear in the submitted code. He ask you to recommend the improved code without your explanation.\n\n'''
}


all_prompt_header_without_comment = {
    1: '''You are given 3 examples. Each example begins with "##Example" and ends with "---". Each example contains the submitted code and the improved code. The submitted code and improved code is written in Java. Your task is to improve your submitted code.\n\n''',
    2: '''You are given 3 examples. Each example begins with "##Example" and ends with "---". Each example contains the submitted code and the improved code. The submitted code and improved code is written in Java.
        
Follow the steps below to improve the given submitted code
step 1 - read the given submitted code in the above examples
step 2 - identify lines that need to be modified, added or deleted in the examples
step 3 - read the given submitted code
step 4 - identify lines that need to be modified, added or deleted
step 5 - generate the improved code without your explanation.\n\n''',
    3: '''You are given 3 examples. Each example begins with "##Example" and ends with "---". Each example contains the submitted code and the improved code. The submitted code and improved code is written in Java.
    
A developer asks you to help him improve his submitted code. He emphasizes that the improved code must have higher quality, conforms to coding convention or standard, and works correctly. He tells you to refrain from putting the submitted code in a class or method, and providing global variables or an implementation of methods that appear in the submitted code. He ask you to recommend the improved code without your explanation.\n\n'''
}




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
        ret = persona_prompt.format(lang)
    else:
        ret = persona_prompt.format('java')

    return ret

def get_prompt_header(lang):

    ret = ''

    if task_name == 'Code_Refinement':
        if dataset_name == 'CodeReviewer-paper':
            ret = all_prompt_header_with_comment[prompt_num].format(lang)
        else:
            ret = all_prompt_header_with_comment[prompt_num].format('java')
    else:
        ret = all_prompt_header_without_comment[prompt_num]

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

        all_model_inputs = []

        if is_use_persona:
            real_persona_prompt = get_persona_prompt(lang)

            all_model_inputs.append({"role": "system", "content": real_persona_prompt})

        prompt = get_prompt_header(lang) + get_sample_str(row) + get_main_prompt(row)
        
        all_model_inputs.append({"role": "user", "content": prompt})

        try:
            resp = client.chat.completions.create(
                model='gpt-3.5-turbo-16k',
                temperature = 0,
                messages= all_model_inputs,
                max_tokens = 512
            )

            generated_text = resp.choices[0].message.content
        
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
