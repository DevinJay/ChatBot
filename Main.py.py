#!/usr/bin/env python
# coding: utf-8

# In[4]:


import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()


# In[5]:


import numpy as np
import tflearn
import tensorflow as tf
import random
import json


# In[6]:


with open("intents.json") as file:
    data=json.load(file)
data


# In[7]:


words=[]
labels=[]
docs_x=[]
docs_y=[]


# In[8]:


nltk.download('punkt')
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds=nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
        if intent["tag"] not in labels:
            labels.append(intent["tag"])


# In[9]:


words=[stemmer.stem(w.lower()) for w in words if w != "?"]
words= sorted(list(set(words)))

labels=sorted(labels)


# In[10]:


training=[]
output=[]


# In[11]:


out_empty=[0 for _ in range(len(labels))]

for x,doc in enumerate(docs_x):
    bag=[]
    wrds=[stemmer.stem(w) for w in doc]
    
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
            
    output_row=out_empty[:]
    output_row[labels.index(docs_y[x])]=1
    
    training.append(bag)
    output.append(output_row)


# In[12]:


training=np.array(training)
output=np.array(output)


# In[14]:


training.shape
output.shape


# In[3]:


#tf.reset_default_graph()

net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)

model=tflearn.DNN(net)


# In[15]:


model.fit(training,output, n_epoch=1000,batch_size=8,show_metric=True)
model.save("model.tflearn")


# In[ ]:


import pickle
with open("data.pickle", "wb") as f:
    pickle.dump((words,labels,training,output),f)

