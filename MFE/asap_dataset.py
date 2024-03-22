import pandas as pd
import os

def load_dataset(filepath, columns):
    asap = pd.read_csv(filepath, sep='\t', encoding='ISO-8859-1', usecols=columns)
    return asap

def load_dataset_5cv(file_path, columns):
    folds_data = []
    for idx in range(0,5):   
        train_data = pd.read_csv(os.path.join(file_path, f'fold_{idx}', f'train_fe.tsv'))
        dev_data = pd.read_csv(os.path.join(file_path, f'fold_{idx}', f'dev_fe.tsv'))
        test_data = pd.read_csv(os.path.join(file_path, f'fold_{idx}', f'test_fe.tsv'))
        
        train_data = scale_dataset(pd.DataFrame(train_data, columns=columns))
        dev_data = scale_dataset(pd.DataFrame(dev_data, columns=columns))
        test_data = scale_dataset(pd.DataFrame(test_data, columns=columns))
        
        folds_data.append((train_data, dev_data, test_data))
    return folds_data

def score_range(prompt):
    match prompt:
        case 1: return (2, 12)
        case 2: return (1, 6)
        case 3 | 4: return (0, 3)
        case 5 | 6: return (0, 4)
        case 7: return (0, 30)
        case 8: return (0, 60)

def min_max_scaler(prompt, score):
    MIN, MAX = score_range(prompt)
    return (score - MIN) / (MAX - MIN)

def scale_to_prompt_range(prompt, score):
    MIN, MAX = score_range(prompt)
    return round(score * (MAX - MIN) + MIN)

def scale_dataset(asap):
    for row in range(len(asap)):
        asap.loc[row, 'nscore'] = min_max_scaler(asap.loc[row, 'essay_set'], asap.loc[row, 'domain1_score'])
    return asap