#!/usr/bin/env python
# coding: utf-8

# In[98]:


import numpy as np 
import pandas as pd 
import nltk 
import string
import re
stop_words = nltk.corpus.stopwords.words('english')
from gensim.models import word2vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from gensim.models import KeyedVectors
import spacy
import wget
import urllib.request
import tensorflow as tf
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.embeddings import Embedding
import contractions
from tensorflow.keras.layers import Dropout
from keras.utils.np_utils import to_categorical 
from tensorflow.keras.layers import Dense,Activation,LSTM,Embedding


# In[4]:


data = pd.read_csv('https://github.com/dipanjanS/nlp_workshop_dhs18/raw/master/Unit%2011%20-%20Sentiment%20Analysis%20-%20Unsupervised%20Learning/movie_reviews.csv.bz2', compression='bz2')
#dataset.info()


# In[6]:


data.info()


# In[15]:


reviews = np.array(data['review'])
sentiments = np.array(data['sentiment'])
stop_words = nltk.corpus.stopwords.words('english')


# In[17]:


# Pre processing 
from bs4 import BeautifulSoup
import tqdm 
import re 
import unicodedata

def strip_html_tags(text):
    soup = BeautifulSoup(text,'html.parser')
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+','\n',stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD',text).encode('ascii','ignore').decode('utf-8','ignore')
    return text


import string
def pre_process_corpus(doc):
    norm_corpus =[]
    for text in tqdm.tqdm(doc):
        text = strip_html_tags(text)
        text = remove_accented_chars(text)
        text = re.sub(r'[^a-zA-Z0-9\s]',' ',text,re.I)
    #subsititute with space for anything other than the combination of [a-zA-Z0-9\s]. this is negated. PLease note tha
        text = text.translate(str.maketrans('','',string.punctuation))
        text = text.lower()
        text = text.strip()
    #Remove spaces at the beginning and at the end of the string:
        text = contractions.fix(text)
    #abbrevations for the words and are not slangs 
    #slangs like ttyl needs to be fixed. however these are abbrevations 
        tokens = nltk.word_tokenize(text)
        filtered_words = [token for token in tokens if token not in stop_words]
        text = " ".join(filtered_words)
        text = re.sub(' +',' ',text)
        norm_corpus.append(text)
    return norm_corpus
    


# In[18]:


normalised_reviews = pre_process_corpus(reviews)


# In[19]:


train_reviews = np.array(normalised_reviews[:35000]) #0th to 35000
train_sentiments = np.array(sentiments[:35000])


test_reviews = np.array(normalised_reviews[35000:]) # 35000 to end 
test_sentiments = np.array(sentiments[35000:])


# In[21]:


train_reviews[0]


# In[22]:


sentiments


# In[24]:


t = Tokenizer()
t.fit_on_texts(train_reviews)


# In[26]:


vocab_size = len(t.word_index)+1
vocab_size


# In[27]:


encoded = t.texts_to_sequences(train_reviews)


# In[36]:


len(encoded)


# In[37]:


import numpy as np


# In[38]:


np.average(len(encoded))


# In[44]:


c= []
for i in range(0,len(encoded)):
    padded_length = len(encoded[i])
    c.append(padded_length)
    


# In[50]:


np.average(c)
#hence pmaking the padded lenght as 150


# In[39]:


type(encoded)


# In[43]:


encoded_array = np.array(encoded)
encoded_array


# In[51]:


max_length  = 150
padded_documents  = pad_sequences(encoded,maxlen=max_length,padding='post')


# In[52]:


f = open("glove.6B.100d.txt",encoding="utf8")
embedding_index = dict()
for line in f:
    values = line.split()
    #spitting by spaces 
    word = values[0]
    #1st value is the word 
    coefs = np.asarray(values[1:],dtype='float32')
    #coefficient are vaues from 1 to end and of type float
    embedding_index[word] = coefs
f.close()
#now the text file is read and the values are stored to the embedding index as dictioniries 


# In[53]:


embedding_index


# In[55]:


embediing_matrix = np.zeros((vocab_size,100))


# In[56]:


for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embediing_matrix[i]=embedding_vector


# In[88]:


target= []
for i in range(0,len(y)):
    if (y[i]=='positive'):
        val = 1
        target.append(val)
    elif (y[i]=='negative'):
        val = 0
        target.append(val)
        
    
        
        


# In[93]:


type(target)

target_array = np.array(target)
target_array


# In[79]:


y = train_sentiments
#y = to_categorical(y)


# In[109]:


model = Sequential()
e = Embedding(vocab_size,100,weights=[embediing_matrix],input_length=150,trainable=False)
model.add(e)
model.add(LSTM(150,return_sequences = True))
model.add(LSTM(100))
model.add(Flatten())
model.add(Dense(50,activation='sigmoid'))
model.add(Dropout(0.2))
# model.add(Dense(25,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(20,activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(10,activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dense(5,activation='sigmoid'))
# model.add(Dropout(0.2))
model.add(Dense(1,activation='softmax'))
model.add(Dropout(0.2))


# In[111]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# In[112]:


model.fit(padded_documents,target_array,epochs=5,verbose=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




