import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import os
from utilities.utils import read_ratings,read_graph_embeddings,read_bert_embedding,matching_Bert_Graph
from models.model3_conf_strategy_att import run_model1,run_model2

graph_embeddings = read_graph_embeddings("embeddings/TRANSDembedding_768.json")
user_bert_embeddings = read_bert_embedding("embeddings/elmo_user_embeddings_nostopw_1024.json")
item_bert_embeddings = read_bert_embedding("embeddings/elmo_embeddings_nostopw_1024.json")

user, item, rating = read_ratings('datasets/movielens/train2id.tsv')
X_graph,X_bert,dim_graph,dim_bert,y = matching_Bert_Graph(user,item,rating,graph_embeddings,user_bert_embeddings,item_bert_embeddings)
model = run_model1(X_graph,X_bert,dim_graph,dim_bert,y,epochs=25,batch_size=1536)


# creates a HDF5 file 'model.h5'
model.save('results/model.h5')