from sklearn.feature_extraction.text import TfidfVectorizer
   
def vectorizer(X_train,  X_val, X_test):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    
    if len(X_val) > 0:
        X_val = vectorizer.transform(X_val)
    
    X_test = vectorizer.transform(X_test)
    return X_train, X_val, X_test