#%%

import json, os, argparse
from collections import Counter

from tqdm import tqdm
import pandas as pd

#%%

double_slash_comment_lang = ['.cs', 'c', 'cpp', 'go', 'java', 'js', 'php']

sharp_comment_lang = ['py', 'rb']

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

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-dataset', default=0, type=int)
arg_parser.add_argument('-task', default=0, type=int)
arg_parser.add_argument('-part', default=0, type=int)
arg_parser.add_argument('-proj', default=0, type=int)
arg_parser.add_argument('--use_persona', action='store_true')


args = arg_parser.parse_args()

datasets = ['TufanoT5', 'D-ACT', 'CodeReviewer']
dataset = datasets[args.dataset]
# dataset = datasets[0]

tasks = ['Code_Refinement', 'Code_Refinement_no_comment']
task = tasks[args.task]
# task = tasks[0]

parts = ['train','valid']
p = parts[args.part]
# p = parts[0]

use_persona = args.use_persona
has_comment = True if task == 'Code_Refinement' else False

if dataset in ['TufanoT5', 'CodeReviewer']:
    input_file_path = '../original-dataset/{}-paper/{}/{}/input.txt'.format(dataset, task, p)
    output_file_path = '../original-dataset/{}-paper/{}/{}/output.txt'.format(dataset, task, p)
    save_dir = '../dataset-for-fine-tuning/{}-paper/{}/{}/'.format(dataset, task, p)
else:
    ## for D-ACT dataset only
    projs = ['android', 'google', 'ovirt']
    proj = projs[args.proj]
    # proj = projs[0]

    input_file_path = '../original-dataset/D-ACT-paper/Code_Refinement_no_comment/dataset-time-wise/{}/{}/input.txt'.format(proj, p)
    input_file_path = '../original-dataset/D-ACT-paper/Code_Refinement_no_comment/dataset-time-wise/{}/{}/output.txt'.format(proj, p)
    save_dir = '../dataset-for-fine-tuning/D-ACT-paper/Code_Refinement_no_comment/dataset-time-wise/{}/{}/'.format(proj,p)


if use_persona:
    save_dir = os.path.join(save_dir, 'dataset-with-persona.jsonl')
else:
    save_dir = os.path.join(save_dir, 'dataset-without-persona.jsonl')


print('input file path:', input_file_path)
print('output file path:', output_file_path)
print('save file dir:', save_dir)
print('use persona?', use_persona)


#%%

def read_file_CodeReviewer_dataset(file_path, has_reviewer_comment = False):
    with open(file_path) as f:
        lines = f.readlines()

    if has_reviewer_comment:
        lines = [s.split('<SEP_DATA>') for s in lines]

        lines = [(s[0].strip().replace('<NEW_LINE>','\n').strip(), s[1].strip()) for s in lines]
        
    else:
        lines = [s.replace('<NEW_LINE>','\n').strip() for s in lines]

    return lines

def read_file_other_dataset(file_path, has_reviewer_comment = False):
    with open(file_path,'r') as f:
        lines = f.readlines()

    if has_reviewer_comment:
        lines = [s.split('<SEP>') for s in lines]

        lines = [(s[0].strip().replace('<NEW_LINE>','\n').strip(), s[1].strip()) for s in lines]

    else:
        lines = [s.replace('<NEW_LINE>','\n').strip() for s in lines]

    return lines


def read_file(dataset, file_path, has_reviewer_comment = False):
    if dataset == 'CodeReviewer':
        lines = read_file_CodeReviewer_dataset(file_path, has_reviewer_comment)
    else:
        lines = read_file_other_dataset(file_path, has_reviewer_comment)

    return lines


def get_example_for_fine_tuning_CodeReviewer_data():
    train_df = pd.DataFrame()
    train_df['input'] = input_code_list
    train_df['output'] = output_code_list
    train_df['lang'] = lang_list

    new_df_by_lang = []

    for lang, sub_df in train_df.groupby('lang'):
        sub_df = sub_df.sample(frac=1.0, random_state=0)

        new_df_by_lang.append(sub_df.head(num_sample))

    new_train_df = pd.concat(new_df_by_lang)

    new_train_df = new_train_df.sample(frac=1.0, random_state=0)

    input_code_list = new_train_df['input'].tolist()
    output_code_list = new_train_df['output'].tolist()
    lang_list = new_train_df['lang'].tolist()

    jsonl_list = []

    system_prompt = 'You are an expert software developer in {}. You always want to improve your code to have higher quality.'

    for input_line, output_line, lang in tqdm(zip(input_code_list, output_code_list, lang_list)):

        final_input_line = ''

        if has_comment:
            final_input_line = final_input_line + input_line[0]

            if lang in double_slash_comment_lang:
                final_input_line = final_input_line + '\n\n' + '// Reviewer comment: ' + input_line[1]
            else:
                final_input_line = final_input_line + '\n\n' + '# Reviewer comment: ' + input_line[1]

        else:
            final_input_line = final_input_line + input_line

        data_dict = []

        if use_persona:
            data_dict.append({
                    'role': 'system',
                    'content': system_prompt
            })

        data_dict.extend([
            {
                'role': 'user',
                'content': final_input_line
            },
            {
                'role': 'assistant',
                'content': output_line
            }
        ])

        jsonl_list.append({'messages': data_dict})

    return jsonl_list


def get_example_for_fine_tuning_other_data():
    jsonl_list = []

    system_prompt = 'You are an expert software developer in Java. You always want to improve your code to have higher quality.'

    for input_line, output_line in tqdm(zip(input_code_list, output_code_list)):

        final_input_line = ''

        if has_comment:
            final_input_line = final_input_line + input_line[0] + '\n\n' + '// Reviewer comment: ' + input_line[1]


        else:
            final_input_line = final_input_line + input_line

        data_dict = []

        if use_persona:
            data_dict.append({
                    'role': 'system',
                    'content': system_prompt
            })

        data_dict.extend([
            {
                'role': 'user',
                'content': final_input_line
            },
            {
                'role': 'assistant',
                'content': output_line
            }
        ])

        jsonl_list.append({'messages': data_dict})

    return jsonl_list


input_code_list = read_file(dataset,input_file_path, has_comment)

output_code_list = read_file(output_file_path, has_comment)


if dataset == 'CodeReviewer':
    if p == 'train':
        num_sample = 1085
    else:
        num_sample = 120
else:
    if p == 'train':
        num_sample = round(0.065*len(input_code_list))
    elif p == 'valid':
        num_sample = round(0.0825*len(output_code_list))
    else:
        num_sample = len(output_code_list)



if dataset == 'CodeReviewer':
    lang_list_dir = '../original-dataset/CodeReviewer-paper/Code_Refinement/{}/lang.txt'.format(p)
    
    with open(lang_list_dir) as f:
        lang_list = f.readlines()

    lang_list = [lang_dict[l.replace('\n','').strip()] for l in lang_list]

    jsonl_list = get_example_for_fine_tuning_CodeReviewer_data()
else:
    jsonl_list = get_example_for_fine_tuning_other_data()


#%%

print('save file to', save_dir)

with open(save_dir, 'w') as outfile:
    for entry in jsonl_list:
        json.dump(entry, outfile)
        outfile.write('\n')
