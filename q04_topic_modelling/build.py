# %load q04_topic_modelling/build.py
# Default imports

import numpy as np
from greyatomlib.nlp_day_02_project.q02_count_vectorizer_for_LDA.build import q02_count_vectorizer_for_LDA
from greyatomlib.nlp_day_02_project.q03_LDA.build import q03_LDA


#  Write your solution here :
def q04_topic_modelling(path,n_top_words=20):
    v1,v2=q02_count_vectorizer_for_LDA(path)
    v3=q03_LDA(path)
    message=[]
    for topic_idx, topic in enumerate(v3):
        message =[v2[i] for i in topic.argsort()[:-n_top_words:-1]]
    return message

# def q04_topic_modelling(path, n_top_words=20):
#     topic_word = q03_LDA(path)
#     topics = []
#     for i, topic_dist in enumerate(topic_word):
#         matrix, feature_names = q02_count_vectorizer_for_LDA(path)
#         topic_words = np.array(feature_names)[np.argsort(topic_dist)][:-n_top_words:-1]
#         func = ('Topic {}: {}'.format(i, ' '.join(topic_words)))
#         topics.append(func)
#     return topics





