#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')


# In[5]:


true_news.head()


# In[6]:


true_news.insert(4, "Label", "true", True)
true_news.head()


# In[7]:


fake_news.insert(4, "Label", "fake", True)
fake_news.head()


# In[8]:


everything_data = true_news.append(fake_news)
everything_data.head()


# In[9]:


everything_data = shuffle(everything_data)
everything_data.head()


# In[10]:


everything_data.reset_index(inplace = True, drop = True)
everything_data.head()


# In[11]:


from sklearn.feature_extraction import text
corpus = everything_data['text']
vectorizer = text.CountVectorizer(binary=True).fit(corpus)
vectorized_text = vectorizer.transform(corpus)


# In[12]:


TfidF = text.TfidfTransformer(norm='l1')
tfidf = TfidF.fit_transform(vectorized_text)


# In[13]:


from sklearn.svm import LinearSVC
labels = everything_data.Label
features = tfidf
model = LinearSVC()
X_train, X_test, y_train, y_test= train_test_split(features, labels, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[17]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[ ]:




