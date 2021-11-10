import nltk
import spacy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import re


stopwords = nltk.corpus.stopwords.words('english')
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
data=pd.read_csv('./amazon_alexa.tsv', sep='\t', header=0)
x=list(data['verified_reviews'])
y=data['rating']
#y=np.where(data['rating']==5,1,0)



def preparing_data(data):
    res_data=[]
    for i in range(0, len(data)):
        review = re.sub('[^a-zA-Z]', ' ', data[i])
        review = review.lower()
        review = review.split()
        review = [WordNetLemmatizer().lemmatize(word) for word in review if not word in set(stopwords)]
        review = ' '.join(review)
        res_data.append(review)
    return res_data

x=preparing_data(x)

x_train, x_test , y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=12
)




tf_idf = TfidfVectorizer()
X_train_tf = tf_idf.fit_transform(x_train)
X_train_tf = tf_idf.transform(x_train)
X_test_tf = tf_idf.transform(x_test)

mdl = LogisticRegression(multi_class='ovr').fit(X_train_tf, y_train)
train_predictions = mdl.predict_proba(X_train_tf)
test_predictions = mdl.predict_proba(X_test_tf)
test_precision = mdl.predict(X_test_tf)
y_test_precision=np.where(np.round(test_precision,0)==y_test,1,0)


multiclass_auc_test1 = roc_auc_score(
    y_test, test_predictions,
    average='macro',
    multi_class='ovr'
)
multiclass_auc_train1 = roc_auc_score(
    y_train, train_predictions,
    average='macro',
    multi_class='ovr'
)
multiclass_auc_test2 = roc_auc_score(
    y_test, test_predictions,
    average='weighted',
    multi_class='ovr'
)
multiclass_auc_train2 = roc_auc_score(
    y_train, train_predictions,
    average='weighted',
    multi_class='ovr'
)


print(f'Logistic regression:\n    Micro precision: {np.sum(y_test_precision)/len(y_test_precision)}')
print(f"    Auc score macro avarage: \n         Train data: {multiclass_auc_train1}\n         Test data:  {multiclass_auc_test1}")
print(f"    Auc score weigh avarage: \n         Train data: {multiclass_auc_train2}\n         Test data:  {multiclass_auc_test2}")

#---------------------------RANDOM_FOREST------------------------------------------------------------------------------------------#
mdl2 = RandomForestRegressor().fit(X_train_tf, y_train)
train_predictions = mdl2.predict(X_train_tf)
test_predictions = mdl2.predict(X_test_tf)
test_results=np.where(np.round(test_predictions,0)==y_test, "Correct", "Incorrect")


y_train_metrics=np.random.choice([0,1], len(y_train))
y_test_metrics=np.random.choice([0,1], len(y_test))

train_predictions_ = np.where(np.round(train_predictions,0) == y_train,
                              np.where(y_train_metrics==0,abs(train_predictions- y_train),abs(1-abs(train_predictions- y_train))) ,
                              np.where(y_train_metrics==0,abs(1-abs(train_predictions- np.round(train_predictions,0))),abs(train_predictions- np.round(train_predictions,0))))
test_predictions_ = np.where(np.round(test_predictions,0) == y_test,
                              np.where(y_test_metrics==0,abs(test_predictions- y_test),abs(1-abs(test_predictions- y_test))) ,
                              np.where(y_test_metrics==0,
                                       abs(1-abs(test_predictions- np.round(test_predictions,0))),
                                       abs(test_predictions- np.round(test_predictions,0))))


train_fpr, train_tpr, _ = roc_curve(y_train_metrics, train_predictions_)
test_fpr, test_tpr, _ = roc_curve(y_test_metrics, test_predictions_)

train_auc, test_auc = np.round(auc(train_fpr, train_tpr), 4), np.round(auc(test_fpr, test_tpr), 4)
file = open('output.txt', 'w')
for i in range(len(test_results)):
    file.writelines('{} : {}\n'.format(i,test_results[i]))

print(f'\n\nRandom forest:\n    Auc score weigh avarage: \n         Train data: {train_auc}\n         Test data:  {test_auc}')

plt.plot(train_fpr, train_tpr, label=f'Train AUC : {train_auc}')
plt.plot(test_fpr, test_tpr, label=f'Test AUC : {test_auc}')
plt.legend()
plt.show()