import pandas as pd

def load_dataset(filepath, columns):
    asap = pd.read_csv(filepath, sep='\t', encoding='ISO-8859-1', usecols=columns)
    return asap

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