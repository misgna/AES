from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from asap_dataset import load_dataset, scale_dataset, scale_to_prompt_range
from features import vectorizer
from models import svr_model, brr_model, xgb_model

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
    train, test = train_test_split(asap[asap['essay_set'] == prompt], test_size=0.2, random_state=42)

    # Vectorize
    train_essay_vec, test_essay_vec = vectorizer(train['essay'].values, test['essay'].values)

    # SVR Prediction
    #scores = svr_model(train_essay_vec, train['nscore'].values, test_essay_vec)
    #scores = brr_model(train_essay_vec, train['nscore'].values, test_essay_vec)
    scores = xgb_model(train_essay_vec, train['nscore'].values, test_essay_vec)

    # Transform score to prompt standard
    y_pred = [scale_to_prompt_range(prompt, score) for score in scores]

    # Compute QWK
    qwk = cohen_kappa_score(test['domain1_score'].values, y_pred, weights='quadratic')
    total_qwk += qwk

    # QWK per prompt
    print(f'QWK for prompt {prompt} is {qwk}')

# Average qwk    
avg_qwk = total_qwk / 8
print(f'Avg qwk: {avg_qwk}')