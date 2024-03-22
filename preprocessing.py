import nltk
import string
import re
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words

from MFE.asap_dataset import scale_dataset
nltk.download('punkt')
nltk.download('words')

def num_chars_essay(text):
    return len(list(text))

def avg_num_chars_word(text):
    return num_chars_essay(text) / num_words_essay(text)

def avg_num_chars_sent(text):
    return num_chars_essay(text) / num_sents_essay(text)

def num_puncts_essay(text):
    return len([char for char in list(text) if char in string.punctuation])

def avg_num_puncts_word(text):
    return num_puncts_essay(text) / num_words_essay(text)

def avg_num_puncts_sent(text):
    return num_puncts_essay(text) / num_sents_essay(text)

def num_words_essay(text):
    return len(word_tokenize(text))

def avg_num_words_sent(text):
    return num_words_essay(text) / num_sents_essay(text)

def num_unique_words_essay(text):
    return len(set(word_tokenize(text)))

def num_sents_essay(text):
    return len(sent_tokenize(text))

def num_numbers_essay(text):
    return len(re.sub("[^0-9]", "", text))

def avg_num_numbers_word(text):
    return num_numbers_essay(text) / num_words_essay(text)

def avg_num_numbers_sent(text):
    return num_numbers_essay(text) / num_sents_essay(text)

def num_correct_words(text):
    # returns correct words
    correct_words = words.words()
    return len(list(set(correct_words) & set(word_tokenize(text)))) / num_words_essay(text)

def num_noun(text):
    pos_tagged = nltk.pos_tag(word_tokenize(text))
    return len([(word, pos) for (word, pos) in pos_tagged if 'NN' in pos])

def num_adj(text):
    pos_tagged = nltk.pos_tag(word_tokenize(text))
    return len([(word, pos) for (word, pos) in pos_tagged if 'JJ' in pos])
    
def num_noun_noun(text):
    pos_tagged = nltk.pos_tag(word_tokenize(text))
    string1 = pos_tagged[:-1]
    string2 = pos_tagged[1:]
    return len([(string1[idx], string2[idx]) for idx in range(len(string1)) if 'NN' in string1[idx][1] and 'NN' in string2[idx][1]])
    
def num_adj_noun(text):
    pos_tagged = nltk.pos_tag(word_tokenize(text))
    string1 = pos_tagged[:-1]
    string2 = pos_tagged[1:]
    return len([(string1[idx], string2[idx]) for idx in range(len(string1)) if 'JJ' in string1[idx][1] and 'NN' in string2[idx][1]])
    

def lexical_features(data):

    asap = pd.DataFrame(data)
    
    asap['num_chars_essay'] = asap['essay'].apply(num_chars_essay)
    asap['avg_num_chars_word'] = asap['essay'].apply(avg_num_chars_word)
    asap['avg_num_chars_sent'] = asap['essay'].apply(avg_num_chars_sent)
    
    asap['num_puncts_essay'] = asap['essay'].apply(num_puncts_essay)
    asap['avg_num_puncts_word'] = asap['essay'].apply(avg_num_puncts_word)
    asap['avg_num_puncts_sent'] = asap['essay'].apply(avg_num_puncts_sent)
    
    asap['num_words_essay'] = asap['essay'].apply(num_words_essay)
    asap['avg_num_words_sent'] = asap['essay'].apply(avg_num_words_sent)
    asap['num_unique_words_essay'] = asap['essay'].apply(num_unique_words_essay)
    
    asap['num_numbers_essay'] = asap['essay'].apply(num_numbers_essay)
    asap['avg_num_numbers_word'] = asap['essay'].apply(avg_num_numbers_word)
    asap['avg_num_numbers_sent'] = asap['essay'].apply(avg_num_numbers_sent)
    
    asap['num_sents_essay'] = asap['essay'].apply(num_sents_essay)
    
    asap['num_correct_words'] = asap['essay'].apply(num_correct_words)
    
    return asap

def load_dataset(filepath, columns):
    folds_data = []
    for idx in range(0,5):   
        train_data = pd.read_csv(os.path.join(filepath, f'fold_{idx}', f'train.tsv'), sep='\t', encoding='ISO-8859-1')
        dev_data = pd.read_csv(os.path.join(filepath, f'fold_{idx}', f'dev.tsv'), sep='\t', encoding='ISO-8859-1')
        test_data = pd.read_csv(os.path.join(filepath, f'fold_{idx}', f'test.tsv'), sep='\t', encoding='ISO-8859-1')

        # Column selections
        columns = ['essay_id', 'essay_set', 'essay', 'domain1_score']
        
        train_data = scale_dataset(pd.DataFrame(train_data, columns=columns))
        train_data['essay'] = train_data['essay'].str.lower()
        dev_data = scale_dataset(pd.DataFrame(dev_data, columns=columns))
        dev_data['essay'] = dev_data['essay'].str.lower()
        test_data = scale_dataset(pd.DataFrame(test_data, columns=columns))
        test_data['essay'] = test_data['essay'].str.lower()
        
        folds_data.append((train_data, dev_data, test_data))
    return folds_data

if __name__ == '__main__':
    filepath = "dataset/asap-data-nea/"
    columns = ['essay_id', 'essay_set', 'essay', 'domain1_score']
    asap = load_dataset(filepath, columns)
    
    for idx, (train, val, test) in enumerate(asap):
        train_fe = lexical_features(train)
        print(f'fold_{idx} train feature engineering completed!')
        val_fe = lexical_features(val)
        print(f'fold_{idx} val feature engineering completed!')
        test_fe = lexical_features(test)
        print(f'fold_{idx} test feature engineering completed!')
        train_fe.to_csv(os.path.join(filepath, f'fold_{idx}', f'train_fe.tsv'), index=False)
        print(f'fold_{idx} file train_fe.tsv is saved')
        val_fe.to_csv(os.path.join(filepath, f'fold_{idx}', f'dev_fe.tsv'), index=False)
        print(f'fold_{idx} file val_fe.tsv is saved')
        test_fe.to_csv(os.path.join(filepath, f'fold_{idx}', f'test_fe.tsv'), index=False)
        print(f'fold_{idx} file test_fe.tsv is saved')