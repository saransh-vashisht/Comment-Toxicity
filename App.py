#!/usr/bin/env python
# coding: utf-8


#  INSTALL DEPENDENCY AND LIBRARIES
# multiple output model
import os # os library help us to work with the different file faucets
import pandas as pd 
import tensorflow as tf
import numpy as np
# tensorflow and tensorflow-gpu  are going to be deep learning models - keras
#  will be used to create sequential model
#  -pandas will help in reading the tabular data 
#  matplotlib - helps for some plotting
#  sklearn
# numpy - numpy is used as np.expand_dims  --> wrap up any of the information inside
# the another set of array ---> used when we got one sample in our batch
# and we want to pass it through our deep learning models bcz we are expecting multiple examples
# in that particular batch so we normally wrap it up inside of that 





# Bringing our data 
df = pd.read_csv( # here we use pd.read_csv function to read the csv
    os.path.join('jigsaw-toxic-comment-classification-challenge','train.csv', 'train.csv')
#     os.path.join ---> gives us the full path to our dataset
)





from tensorflow.keras.layers import TextVectorization
# textvectorization is used for tokenization





X = df['comment_text']
y = df[df.columns[2:]].values
# .values convert it into numpy array





MAX_FEATURES = 200000 # number of words in the vocab





vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
# output_sequence_length ---> maximum length of the sentence in our token 
# output_mode in the form of integer


# In[7]:


vectorizer.adapt(X.values)
# adapt will help us to learn all the words in vocabulary 
# we use  X.values cuz we need numpy array instead of pandas series
# vectorizer.get_vocabulary()  ---> can also be used to get the vocabulary 





vectorized_text = vectorizer(X.values)
#  here we will pass all our x values through the vectorizer and we gonna get the vectorized_text data





import tensorflow as tf
import gradio as gr





# model.save('toxicity.h5')





model = tf.keras.models.load_model('toxicity.h5')





def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text





interface = gr.Interface(fn=score_comment,  capture_session=True,
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')





interface.launch(share=True)







