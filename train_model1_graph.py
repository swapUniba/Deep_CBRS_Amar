import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os
from utilities.utils import read_graph_embeddings,read_ratings,matching_graph_emb_id
from models.model1 import run_model

ent_embeddings = read_graph_embeddings('embeddings/TRANSEembedding_768.json')
user, item, rating = read_ratings('datasets/dbbook/train2id.tsv')
X, y, dim_embeddings = matching_graph_emb_id(user,item,rating,ent_embeddings)

print("Embedding dimension: ",dim_embeddings)
model = run_model(X,y,dim_embeddings,epochs=25,batch_size=512)

# creates a HDF5 file 'model.h5'
model.save('results/model.h5')



