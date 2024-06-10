#%%

## adapt code from https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer/code/evaluator/CodeBLEU

import os, sys
import pandas as pd
from CodeBLEU.calc_code_bleu import get_codebleu

#%%

dataset_names = ['TufanoT5-paper', 'D-ACT-paper']

task_names = ['with-comment', 'without-comment']

projs = ['android','google','ovirt']


main_data_dir = '../../datasets/original-dataset/'

data_paths = {
    'TufanoT5-paper': {
        'with-comment': os.path.join(main_data_dir,'TufanoT5-paper/with-comment/test/input.txt'),
        'without-comment': os.path.join(main_data_dir,'TufanoT5-paper/without-comment/test/input.txt')
    },
    'D-ACT-paper': {
        'without-comment': os.path.join(main_data_dir,'D-ACT-paper/{}/test/input.txt')
    }
}


dataset_name = dataset_names[int(sys.argv[1])]
task_name = task_names[int(sys.argv[2])]

result_path = sys.argv[3]

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

try:
    if dataset_name == 'D-ACT-paper':
        ## add sub-folder here
        proj = projs[int(sys.argv[4])]

        gt_file_path = data_paths[dataset_name][task_name].format(proj)

        gt_file_path = '../../original-dataset/D-ACT-paper/Code_Refinement_no_comment/dataset-time-wise/{}/test/output.txt'.format(proj)

        eval_file_dir = '../../evaluation_result/CodeBLEU/D-ACT-paper/Code_Refinement_no_comment/dataset-time-wise/{}'.format(proj)
        
    else:

        gt_file_path = data_paths[dataset_name][task_name]

        eval_file_dir = '../../evaluation_result/CodeBLEU/{}/{}/'.format(dataset_name, task_name)

            
    os.makedirs(eval_file_dir, exist_ok = True)

    print('ground truth file path:', gt_file_path)
    print('evaluating output from file:', result_path)

    generated_outputs = get_output_from_file(result_path)

    with open(gt_file_path) as f:
        ground_truth = f.readlines()
        ground_truth = [l.strip() for l in ground_truth]


    pred_results = get_codebleu(ground_truth, generated_outputs, 'java')

    print('CodeBLEU:', round(pred_results*100,2))

    print('-'*30)


except Exception as e:
    print(e)
    print('-'*30)
    