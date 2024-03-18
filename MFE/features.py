import nltk
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import words

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
    #asap['avg_num_chars_word'] = asap['essay'].apply(avg_num_chars_word)
    #asap['avg_num_chars_sent'] = asap['essay'].apply(avg_num_chars_sent)
    
    asap['num_puncts_essay'] = asap['essay'].apply(num_puncts_essay)
    #asap['avg_num_puncts_word'] = asap['essay'].apply(avg_num_puncts_word)
    #asap['avg_num_puncts_sent'] = asap['essay'].apply(avg_num_puncts_sent)
    
    asap['num_words_essay'] = asap['essay'].apply(num_words_essay)
    #asap['avg_num_words_sent'] = asap['essay'].apply(avg_num_words_sent)
    asap['num_unique_words_essay'] = asap['essay'].apply(num_unique_words_essay)
    
    asap['num_numbers_essay'] = asap['essay'].apply(num_numbers_essay)
    #asap['avg_num_numbers_word'] = asap['essay'].apply(avg_num_numbers_word)
    #asap['avg_num_numbers_sent'] = asap['essay'].apply(avg_num_numbers_sent)
    
    asap['num_sents_essay'] = asap['essay'].apply(num_sents_essay)
    
    #asap['num_correct_words'] = asap['essay'].apply(num_correct_words)
    
    
    return asap
   
def vectorizer(X_train,  X_test):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    '''
    if len(X_val) > 0:
        X_val = vectorizer.transform(X_val)
    '''
    X_test = vectorizer.transform(X_test)
    return X_train, X_test