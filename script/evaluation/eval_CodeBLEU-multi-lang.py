#%%

## adapt code from https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer/code/evaluator/CodeBLEU

import os, sys
import pandas as pd
from CodeBLEU.calc_code_bleu import get_codebleu
import numpy as np

#%%


dataset_name = 'CodeReviewer-paper'
task_name = 'Code_Refinement'


result_path = sys.argv[1]


def get_output_from_df(result_path):
    df = pd.read_csv(result_path)

    df = df.fillna(' ')

    generated_outputs = df['extracted_output'].tolist()

    return generated_outputs


def get_output_from_text_file(result_path):
    with open(result_path) as f:
        generated_outputs = f.readlines()

    generated_outputs = [l.strip() for l in generated_outputs]

    return generated_outputs

def get_output_from_file(result_path):
    file_ext = result_path.split('.')[-1]

    if file_ext == 'txt':
        print('get output from text file')
        return get_output_from_text_file(result_path)
    elif file_ext == 'csv':
        print('get output from csv file')
        return get_output_from_df(result_path)
    else:
        print('incorrect file extension')
        exit()


#%%


lang_dict = {'go': 'go',
    'php': 'php',
    '.cs': 'c_sharp',
    'java': 'java',
    'js': 'javascript',
    'c': 'c',
    'cpp': 'cpp',
    'rb': 'ruby',
    'py': 'python'}



gt_file_path = '../../datasets/original-dataset/CodeReviewer-paper/test/output.txt'

eval_file_dir = '../../evaluation_result/CodeBLEU/{}/{}/'.format(dataset_name, task_name)

os.makedirs(eval_file_dir, exist_ok = True)


print('ground truth file path:', gt_file_path)
print('evaluating output from file:', result_path)

generated_outputs = get_output_from_file(result_path)

with open(gt_file_path) as f:
    ground_truth = f.readlines()
    ground_truth = [l.strip() for l in ground_truth]


#%%

with open('../../datasets/original-dataset/CodeReviewer-paper/test/lang.txt') as f:
    lang_list = f.readlines()

lang_list = [s.replace('\n','').strip() for s in lang_list]

df = pd.DataFrame()
df['gt'] = ground_truth
df['output'] = generated_outputs
df['lang'] = lang_list



#%%

codebleu_by_lang = {}


for name, sub_df in df.groupby('lang'):
    sub_ground_truth = sub_df['gt'].tolist()
    sub_generated_outputs = sub_df['output'].tolist()
    lang = sub_df['lang'].tolist()[0]

    pred_results = get_codebleu(ground_truth, generated_outputs, lang_dict[lang])

    codebleu_by_lang[lang] = pred_results


print('avg codeBLEU from all lang:', round(np.mean(list(codebleu_by_lang.values()))*100,2))

print(codebleu_by_lang)
