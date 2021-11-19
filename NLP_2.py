import re
import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


def clear_text(start_data):
    stopwords = nltk.corpus.stopwords.words('english')
    review = re.sub('[^a-zA-Z]', ' ', start_data)
    review = review.lower()
    review = review.split()
    review = [WordNetLemmatizer().lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    return review

def preparing_data(data):
    res_data=[]
    for i in range(0, len(data)):
        res_data.append(clear_text(data[i]))
    return res_data


def get_corpus():
    data = pd.read_csv('./amazon_alexa.tsv', sep='\t', header=0)
    x = {"text":preparing_data(list(data['verified_reviews'])), "rating" : data['rating']}
    inp_=['All excellent. I am completely satisfied']
    inp = preparing_data(inp_)
    dt=pd.DataFrame(data=x)
    data_ = dt['text']
    print(data_)


    tf_idf = TfidfVectorizer()
    vectors = tf_idf.fit_transform(data_)
    vector_inp = tf_idf.transform(inp)

    nn = NearestNeighbors(metric='cosine')
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vector_inp, n_neighbors=15)
    neighbors = pd.DataFrame({'distance': distances.flatten(), 'id': indices.flatten()})

    nearest_info = (
        dt.merge(neighbors, right_on='id', left_index=True).sort_values('distance')[['text','distance']])

    dt1=pd.DataFrame(data={'text':pd.unique(nearest_info['text']),'distance' : pd.unique(nearest_info['distance'])})
    print('test: ',inp[0])
    print(dt1)



get_corpus()
