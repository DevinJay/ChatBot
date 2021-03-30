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
import pickle


# In[6]:


with open("intents.json") as file:
    data=json.load(file)



# In[7]:

try:
	
	with open("data.pickle", "rb") as f:
		words,labels,training,output=pickle.load(f)

except:

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





	training=np.array(training)
	output=np.array(output)

	
	with open("data.pickle", "wb") as f:
		pickle.dump((words,labels,training,output), f)

# In[3]:


#tf.reset_default_graph()

net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)

model=tflearn.DNN(net)


# In[15]:

try:
	
	model.load("model.tflearn")
# 	1
except:
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")

def bag_of_words(s,words):
	bag=[0 for _ in range(len(words))]
	
	s_words=nltk.word_tokenize(s)
	s_words=[stemmer.stem(word.lower()) for word in s_words ]

	for se in s_words:
		for i,w in enumerate(words):
			if w== se:
				bag[i]=1

	return np.array(bag)

def chat():
	print("\n\nStart talking with the bot! (type quit to stop)\n\n")
	while True:
		inp= input("You: ")
		if inp.lower() == "quit":
			break

		results=model.predict([bag_of_words(inp,words)])
		#print(results)
		results_index=np.argmax(results)
		tag= labels[results_index]
		#print(tag)


		for tg in data["intents"]:
			if tg['tag']== tag:
				responses= tg['responses']
		print('\n')
		print('Bot: '+random.choice(responses))
		print('\n')
chat()




