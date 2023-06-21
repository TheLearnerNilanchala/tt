import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'F:\Class Assignments_Notes\June_Month\19th\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(x_train, y_train)
bias

variance = classifier.score(x_test, y_test)
variance

from sklearn.svm import SVC
s_v_m = SVC()
s_v_m.fit(x_train,y_train)

y_pred = s_v_m.predict(x_test)

from sklearn.metrics import accuracy_score
kn=accuracy_score(y_test, y_pred)
print(kn)

from sklearn.metrics import confusion_matrix
classifier_svm = confusion_matrix(y_test, y_pred)
print(classifier_svm)

from sklearn.metrics import classification_report
svm_report = classification_report(y_test, y_pred)
print(svm_report)

bias = s_v_m.score(x_train, y_train)
bias

variance = s_v_m.score(x_test, y_test)
variance


from sklearn.neighbors import KNeighborsClassifier
neighbours = KNeighborsClassifier()
neighbours.fit(x_train, y_train)

y_pred = neighbours.predict(x_test)

from sklearn.metrics import accuracy_score
nh = accuracy_score(y_test, y_pred)
print(nh)

from sklearn.metrics import confusion_matrix
hm = confusion_matrix(y_test, y_pred)
print(hm)

bias = neighbours.score(x_train, y_train)
bias

variance = neighbours.score(x_test, y_test)
variance

from sklearn.tree import DecisionTreeClassifier
Tree = DecisionTreeClassifier()
Tree.fit(x_train, y_train)

y_pred = Tree.predict(x_test)

from sklearn.metrics import accuracy_score
ac_tree = accuracy_score(y_test, y_pred)
print(ac_tree)

bias = Tree.score(x_train, y_train)
bias

variance = Tree.score(x_test, y_test)
variance

from sklearn.naive_bayes import GaussianNB
naivebayes = GaussianNB()
naivebayes.fit(x_train,y_train)

y_pred = naivebayes.predict(x_test)

from sklearn.metrics import accuracy_score
ac_NB = accuracy_score(y_test, y_pred)
print(ac_NB)

bias = naivebayes.score(x_train, y_train)
bias

variance = naivebayes.score(x_test, y_test)
variance

from xgboost import XGBClassifier
NLP = XGBClassifier()
NLP.fit(x_train, y_train)

y_pred = NLP.predict(x_test)

from sklearn.metrics import accuracy_score
ac_XGB = accuracy_score(y_test, y_pred)

bias = NLP.score(x_train, y_train)
bias

variance = NLP.score(x_test, y_test)
variance

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
ac

bias = classifier.score(x_train, y_train)
bias 

varience = classifier.score(x_test, y_test)
varience

from lazypredict.Supervised import LazyClassifier
lc = LazyClassifier()
lc.fit(x_train,x_test, y_train, y_test)



