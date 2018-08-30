# %load q01_load_data_tfidf/build.py
# Default imports

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Write your solution here :

def q01_load_data_tfidf(path,max_df=0.95,min_df=2,no_features=1000):
    variable1=pd.read_csv(path)
    tf_vect=TfidfVectorizer(max_df=max_df,min_df=min_df,max_features=no_features,stop_words='english')
    variable2=tf_vect.fit_transform(variable1['talkTitle'])
    variable3=tf_vect.get_feature_names()
    return variable1,variable2,variable3

# def q01_load_data_tfidf(path, max_df=0.95, min_df=2, no_features=1000):
#     dataset = pd.read_csv(path)
#     tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=no_features, stop_words='english')
#     tfidf = tfidf_vectorizer.fit_transform(dataset['talkTitle'])
#     tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#     return dataset, tfidf, tfidf_feature_names



