import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os
from utilities.utils import read_ratings,read_graph_embeddings,read_bert_embedding,matching_userBert_itemGraph,matching_userGraph_itemBert
from models.model1 import run_model

user_embeddings = read_graph_embeddings("embeddings/TRANSHembedding_768.json")
item_embeddings = read_bert_embedding("embeddings/ITEM_embeddingslastlayer.json")
user, item, rating = read_ratings('datasets/dbbook/train2id.tsv')
X, y, dim_embeddings = matching_userGraph_itemBert(user,item,rating,user_embeddings,item_embeddings)
model = run_model(X,y,dim_embeddings,epochs=25,batch_size=512)

# creates a HDF5 file 'model.h5'
model.save('results/model.h5')



