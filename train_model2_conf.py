import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os
from utilities.utils import read_ratings,read_graph_embeddings,read_bert_embedding,matching_Bert_Graph_conf
from models.model2_conf import run_conf_1,run_conf_2

graph_embeddings = read_graph_embeddings("embeddings/TRANSDembedding_768.json")
user_bert_embeddings = read_bert_embedding("embeddings/UserProfiles_lastLayer.json")
item_bert_embeddings = read_bert_embedding("embeddings/ITEM_embeddingslastlayer.json")

user, item, rating = read_ratings('datasets/dbbook/train2id.tsv')
X, y, dim_embeddings = matching_Bert_Graph_conf(user,item,rating,graph_embeddings,user_bert_embeddings,item_bert_embeddings)
model = run_conf_2(X,y,dim_embeddings,epochs=30,batch_size=512)

# creates a HDF5 file 'model.h5'
model.save('results/model.h5')



