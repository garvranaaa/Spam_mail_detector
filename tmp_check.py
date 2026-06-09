import pandas as pd
import re
import nltk
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words=set(stopwords.words('english'))
stemmer=PorterStemmer()

def preprocess(text):
    text=re.sub(r'\W',' ',str(text))
    text=text.lower()
    words=[stemmer.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

path=Path('spam.csv')
df=pd.read_csv(path, encoding='latin-1', usecols=['v1','v2'])
df.columns=['label','message']
df=df.dropna()
df['label']=df['label'].map({'ham':0,'spam':1})
df=df[df['label'].isin([0,1])]
df['cleaned']=df['message'].apply(preprocess)
vectorizer=TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X=vectorizer.fit_transform(df['cleaned'])
y=df['label']
print('spam ratio', y.mean())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model=LogisticRegression(max_iter=1000,C=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print('accuracy',accuracy_score(y_test,y_pred))
print('precision',precision_score(y_test,y_pred))
print('recall',recall_score(y_test,y_pred))
print('f1',f1_score(y_test,y_pred))
example='sdhasjhdlashdklashkldhsakldhklashdlk please buy this soon offer ending'
clean=preprocess(example)
vec=vectorizer.transform([clean])
print('clean', clean)
print('vocab intersects', [w for w in clean.split() if w in vectorizer.vocabulary_])
print('prob spam', model.predict_proba(vec)[0][1])
print('pred', model.predict(vec)[0])
print('top features sample', [vectorizer.get_feature_names_out()[i] for i in vec.nonzero()[1][:50]])
