import torch
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import vocab
import pandas as pd

from collections import OrderedDict
from asap_dataset import ASAPDataset, collate_fn

tokenizer = get_tokenizer('basic_english')

essay_set = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}
def min_max_normalization(score, prompt):
    """
    Normalizes the score into the range from 0 to 1
    parameters: score, prompt
    return: normalized score [0 - 1]
    """

    return (score - essay_set[prompt][0]) / (essay_set[prompt][1] - essay_set[prompt][0])

def scaler(score, prompt):
    """
    Transform the score into the range of a prompt's score range.
    parameters: score, prompt
    return: an integer, bounded to the prompt's score range
    """

    return round(score * (essay_set[prompt][1] - essay_set[prompt][0]) + essay_set[prompt][0])

def split_dataset(prompt, asap): 
    """
    Split the dataset into training, validation and test data with 60%, 20%, and 20%, respectively.
    parameters: prompt and dataset (asap)
    return: sample of the datast as training, validation and test data 
    """

    train, val, test = torch.utils.data.random_split(asap[asap['essay_set']==prompt].values, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42))
    return train, val, test

def essay_vectorizer(text):
    """
    parameter: essay text
    return: essay vector
    """
    tokenized_essay = [tokenizer(field[2].lower()) for field in text]

    tokens = set()
    for essay in tokenized_essay:
        tokens.update(essay)
    
    tokens = list(tokens)

    vocab_essay = vocab(OrderedDict([(token, 1) for token in tokens]), specials=['<unk>'])
    vocab_essay.set_default_index(vocab_essay['<unk>'])

    return vocab_essay

def essay_dataloader(prompt, batch_size):
    """
    The dataset can be download from https://www.kaggle.com/competitions/asap-aes.
    Features, like essay set, text (essay) and score (domain1_score) are filtered

    parameters: prompt, batch_size
    return: Dataset adjusted into batches and vocab of the training dataset 
    """
    
    file_path = '../dataset/asap-aes/training_set_rel3.tsv'
    columns = ['essay_id', 'essay_set', 'essay', 'domain1_score']
    asap = pd.read_csv(file_path, sep='\t', encoding='ISO-8859-1', usecols=columns)

    train, val, test = split_dataset(prompt, asap)
    vocab_essay = essay_vectorizer(train)

    asap_train = ASAPDataset(train, vocab_essay)
    train_dl = DataLoader(asap_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    asap_val = ASAPDataset(val, vocab_essay)
    val_dl = DataLoader(asap_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    asap_test = ASAPDataset(test, vocab_essay)
    test_dl = DataLoader(asap_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dl, val_dl, test_dl, vocab_essay