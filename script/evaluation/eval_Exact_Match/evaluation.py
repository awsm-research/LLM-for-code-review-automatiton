import re, sys, os, time

from smooth_bleu import bleu_fromstr, my_bleu_fromstr

import pandas as pd
import numpy as np

def remove_diff_symbol(code):
    code_lines = code.splitlines()

    new_code_lines = []

    for l in code_lines:
        if l.startswith('+') or l.startswith('-'):
            new_code_lines.append(l[1:])
        else:
            new_code_lines.append(l)

    return '\n'.join(new_code_lines)

def remove_comments(code):

    # pattern = r'(/\*.*?\*/|//.*?$|\".*?\")'
    pattern = r'/\*.*?\*/|//.*?$'
    tmp_code = re.sub(pattern, '', code, flags=re.DOTALL|re.MULTILINE)
    pattern = r'(?m)^\s*#.*?$'
    return re.sub(pattern, '', tmp_code)

def get_em_trim(gold, pred):
    gold_lines = gold.split("\n")
    pred_lines = pred.split("\n")
    jumps = [0]
    for line in pred_lines:
        jumps.append(len(line)+jumps[-1])
    gold_words = []
    pred_words = []
    for line in gold_lines:
        gold_words.extend(line.split())
    for line in pred_lines:
        pred_words.extend(line.split())
    em_trim = 0
    if len(pred_words) >= len(gold_words):
        for jump in jumps:
            if jump+len(gold_words) > len(pred_words):
                break
            if pred_words[jump:jump+len(gold_words)] == gold_words:
                em_trim = 1
                break
        # for i in range(len(pred_words)-len(gold_words)+1):
        #     if pred_words[i:i+len(gold_words)] == gold_words:
        #         em_trim = 1
        #         break
    return em_trim


def get_em_no_space(gold, pred):
    gold_lines = gold.split("\n")
    pred_lines = pred.split("\n")
    gold_line_no_space = [re.sub(r'\s', '', line) for line in gold_lines]
    pred_line_no_space = [re.sub(r'\s', '', line) for line in pred_lines]
    jumps = [0]
    for line in pred_line_no_space:
        jumps.append(len(line)+jumps[-1])
    gold_string_no_space = "".join(gold_line_no_space)
    pred_string_no_space = "".join(pred_line_no_space)
    em_no_space = 0
    if len(pred_string_no_space) >= len(gold_string_no_space):
        for jump in jumps:
            if jump+len(gold_string_no_space) > len(pred_string_no_space):
                break
            if pred_string_no_space[jump:jump+len(gold_string_no_space)] == gold_string_no_space:
                em_no_space = 1
                break
    return em_no_space


def get_em_no_comment(gold, pred):
    gold_no_comment = remove_comments(gold)
    pred_no_comment = remove_comments(pred)
    return get_em_no_space(gold_no_comment, pred_no_comment)


def get_em(gold, pred):
    gold_lines = gold.split("\n")
    pred_lines = pred.split("\n")
    gold_words = []
    pred_words = []
    for line in gold_lines:
        gold_words.extend(line.split())
    for line in pred_lines:
        pred_words.extend(line.split())
    em = 0
    if pred_words == gold_words:
        em = 1
    return em


def myeval(gold, pred):
    em = get_em(gold, pred)
    em_trim = get_em_trim(gold, pred)
    em_no_space = get_em_no_space(gold, pred)
    em_no_comment = get_em_no_comment(gold, pred)
    
    return em, em_trim, em_no_space, em_no_comment



dataset_names = ['CodeReviewer-paper', 'TufanoT5-paper', 'D-ACT-paper']

task_names = ['Code_Refinement', 'Code_Refinement_no_comment']


projs = ['android','google','ovirt']


dataset_name = dataset_names[int(sys.argv[1])]
task_name = task_names[int(sys.argv[2])]

result_path = sys.argv[3]

main_data_dir = '../datasets/original-dataset/'

input_paths = {
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


output_paths = {
    'CodeReviewer-paper': {
        'with-comment': os.path.join(main_data_dir,'CodeReviewer-paper/test/output.txt')
    },
    'TufanoT5-paper': {
        'with-comment': os.path.join(main_data_dir,'TufanoT5-paper/with-comment/test/output.txt'),
        'without-comment': os.path.join(main_data_dir,'TufanoT5-paper/without-comment/test/output.txt')
    },
    'D-ACT-paper': {
        'without-comment': os.path.join(main_data_dir,'D-ACT-paper/{}/test/output.txt')
    }
}


def get_output_from_df(result_path):
    df = pd.read_csv(result_path)

    df = df.fillna(' ')


    if 'D-ACT' in result_path:
        generated_outputs = df['output'].tolist()
    else:
        generated_outputs = df['extracted_output'].tolist()

    return generated_outputs


def get_output_from_text_file(result_path):
    with open(result_path) as f:
        generated_outputs = f.readlines()

    generated_outputs = [l.strip() for l in generated_outputs]

    if dataset_name == 'CodeReviewer':
        generated_outputs = [s.replace('<add>','') for s in generated_outputs]

    elif dataset_name == 'TufanoT5':
        generated_outputs = [s.replace('<NEW_LINE>',' \n') for s in generated_outputs]

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


def show_em_percent(em_list, em_label):
    total_em = np.sum(em_list)
    total_test_set = len(em_list)

    em_percent = (total_em/total_test_set)*100

    print('{}: {}/{} or {} %'.format(em_label, total_em, total_test_set, em_percent))
    # print(em_label, total_em, em_percent)
    print('-'*30)



try:
    if dataset_name == 'D-ACT-paper':
        ## add sub-folder here

        proj = projs[int(sys.argv[4])]

        gt_file_path = output_paths[dataset_name][task_name].format(proj)
        

    else:
        gt_file_path = output_paths[dataset_name][task_name]


       
    print('ground truth file path:', gt_file_path)
    print('evaluating output from file:', result_path)

    generated_outputs = get_output_from_file(result_path)

    with open(gt_file_path) as f:
        ground_truth = f.readlines()
        ground_truth = [l.replace("<NEW_LINE>",'\n').strip() for l in ground_truth]


    all_em, all_em_trim, all_em_no_space, all_em_no_comment = [], [], [], []

    start = time.time()

    for gold, pred in zip(ground_truth, generated_outputs):

        em, em_trim, em_no_space, em_no_comment = myeval(remove_diff_symbol(gold), pred)

        all_em.append(em)
        all_em_trim.append(em_trim)
        all_em_no_space.append(em_no_space)
        all_em_no_comment.append(em_no_comment)

    end = time.time()

    show_em_percent(all_em_no_space, 'EM no space')

    result_df = pd.DataFrame()
    result_df['em'] = all_em
    result_df['em_trim'] = all_em_trim
    result_df['em_no_space'] = all_em_no_space
    result_df['em_no_comment'] = all_em_no_comment


    print('time spent: {} secs'.format(end - start))

    print('-'*30)
        

except Exception as e:
    print(e)