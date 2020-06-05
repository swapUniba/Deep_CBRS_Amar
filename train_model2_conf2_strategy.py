import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os
from utilities.utils import read_ratings,read_graph_embeddings,read_bert_embedding,matching_Bert_Graph
from models.model2_conf2_strategy import run_model

graph_embeddings = read_graph_embeddings("embeddings/DISTMULT_512.json")
user_bert_embeddings = read_bert_embedding("embeddings/UserProfiles_lastLayer.json")
item_bert_embeddings = read_bert_embedding("embeddings/ITEM_embeddingslastlayer.json")

user, item, rating = read_ratings('datasets/dbbook/train2id.tsv')
X_graph,X_bert,dim_graph,dim_bert,y = matching_Bert_Graph(user,item,rating,graph_embeddings,user_bert_embeddings,item_bert_embeddings)
model = run_model(X_graph,X_bert,dim_graph,dim_bert,y,epochs=25,batch_size=512)


# creates a HDF5 file 'model.h5'
model.save('results/model.h5')



