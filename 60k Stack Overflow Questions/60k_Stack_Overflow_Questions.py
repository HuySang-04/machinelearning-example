#https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import  Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from lazypredict.Supervised import LazyClassifier
from sklearn import svm

data = pd.read_csv('/home/sang/BaiTap/MachineLearning/60k Stack Overflow Questions/train.csv')

# print(data.shape)
# print(data.head())
# print(data.describe())
# print(data.info())
# print(data.isnull().sum())

# plt.figure(figsize=(8, 6))
# sns.countplot(x=data['Y'])
# plt.title('Distribution of target')
# plt.xlabel('Target')
# plt.show()

x = data.drop(['Id', 'CreationDate', 'Y'], axis=1)
y = data['Y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
# models,predictions = clf.fit(x_train, x_test, y_train, y_test)

preprocessor = ColumnTransformer(transformers=[
    ('vectorizer_title', TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2), min_df=0.01, max_df=0.95), 'Title'),
    ('vectorizer_body', TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2), min_df=0.01, max_df=0.95), 'Body'),
    ('vectorizer_tags', TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2), min_df=0.01, max_df=0.95), 'Tags')
], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kbest', SelectKBest(chi2, k=100)),
    # ('cls', RandomForestClassifier(random_state=42, n_estimators=10))
    ('cls', svm.SVC())
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))

