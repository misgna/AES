from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import scipy
import torch

from asap_dataset import load_dataset, scale_dataset, scale_to_prompt_range, load_dataset_5cv
from features import vectorizer
from models import svr_model, brr_model, xgb_model

def train_80_20(model):
    # Load dataset
    # Dataset can be downloaded from https://www.kaggle.com/competitions/asap-aes
    filepath = '../dataset/asap-aes/training_set_rel3.tsv'
    columns = ['essay_id', 'essay_set', 'essay', 'domain1_score']
    asap = load_dataset(filepath, columns)
    asap = scale_dataset(asap) # add a normalized score column

    total_qwk = 0.0
    # Prompt-specific models
    for prompt in range(1, 9):

        # Split dataset for training and test
        train, test = train_test_split(asap[asap['essay_set'] == prompt], test_size=0.33, random_state=42)

        # Transform score to prompt standard
        scores = prediction_scores(train, test, test, model, prompt)

        # Compute QWK
        qwk = cohen_kappa_score(test['domain1_score'].values, scores, weights='quadratic')
        total_qwk += qwk

        # QWK per prompt
        print(f'QWK for prompt {prompt} is {qwk}')

    # Average qwk    
    avg_qwk = total_qwk / 8
    print(f'Avg qwk: {avg_qwk}')
    
def train_5cv(model):
    # Load dataset
    # Model testing using the 5-fold cv used by Taghipour and Ng (2016)
    filepath = '../dataset/asap-data-nea'
    columns = ['essay_id', 'essay_set', 'essay', 'domain1_score', 'num_chars_essay', 'num_puncts_essay', 'num_words_essay', 'num_unique_words_essay', 'num_numbers_essay', 'num_sents_essay']
    asap_folds = load_dataset_5cv(filepath, columns)

    qwks = []
    for idx, (train, val, test) in enumerate(asap_folds):
        prompt_qwk = []
        for prompt in range(1, 9):
            train_prompt = train[train['essay_set'] == prompt]
            test_prompt = test[test['essay_set'] == prompt]
            val_prompt = val[val['essay_set'] == prompt]
            scores = prediction_scores(train_prompt, val_prompt, test_prompt, model, prompt)

            qwk = cohen_kappa_score(test_prompt['domain1_score'].values, scores, weights='quadratic')
            
            prompt_qwk.append(qwk)

        qwks.append(prompt_qwk)
    qwks = torch.as_tensor(qwks)
    fold_avg = torch.mean(qwks, 0)
    fold_max = torch.max(qwks, 0).values
    print(f'Avg qwks of avg folds {torch.mean(fold_avg)}')
    print(f'Avg qwks of max folds {torch.mean(fold_max)}')
        
    

def prediction_scores(train, val, test, model, prompt):
    # Vectorize
    train_essay_vec, val_essay_vec, test_essay_vec = vectorizer(train['essay'].values, val['essay'].values, test['essay'].values)
    train_new, val_new, test_new = [], [], []

    scaler = MinMaxScaler()

    if model not in ['svr', 'brr', 'xgb']:
        # Lexical features
        columns = ['num_chars_essay', 'num_puncts_essay', 'num_words_essay', 'num_unique_words_essay', 'num_numbers_essay', 'num_sents_essay']
        train_features = pd.DataFrame(train, columns=columns)
        train_features[columns] = scaler.fit_transform(train_features[columns])

        val_features = pd.DataFrame(val, columns=columns)
        val_features[columns] = scaler.fit_transform(val_features[columns])

        test_features = pd.DataFrame(test, columns=columns)
        test_features[columns] = scaler.fit_transform(test_features[columns])

        #Merge features
        train_new = scipy.sparse.hstack([train_essay_vec, train_features])
        val_new = scipy.sparse.hstack([val_essay_vec, val_features])
        test_new = scipy.sparse.hstack([test_essay_vec, test_features])

    # Prediction models
    scores = []
    match model:
        case 'svr': scores = svr_model(train_essay_vec, train['nscore'].values, test_essay_vec)
        case 'brr': scores = brr_model(train_essay_vec, train['nscore'].values, test_essay_vec)
        case 'xgb': scores = xgb_model(train_essay_vec, train['nscore'].values, test_essay_vec)
        case 'svr_fe': scores = svr_model(train_new, train['nscore'].values, test_new)
        case 'brr_fe': scores = brr_model(train_new, train['nscore'].values, test_new)
        case 'xgb_fe': scores = xgb_model(train_new, train['nscore'].values, test_new)


    # Transform score to prompt standard
    return [scale_to_prompt_range(prompt, score) for score in scores]

model = 'svr_fe'
partition = '5cv'

if partition == 'simple':
    train_80_20(model)
elif partition == '5cv':
    train_5cv(model)


   






