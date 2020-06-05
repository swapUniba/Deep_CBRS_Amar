import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#CONFIGURAZIONE 1
def run_conf_1(X,y,dim_embeddings,epochs,batch_size):
  model = keras.Sequential()
  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))
  x1_user = keras.layers.Dense(8, activation=tf.nn.relu)(input_users_1)
  x1_item = keras.layers.Dense(8, activation=tf.nn.relu)(input_items_1)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))
  x2_user = keras.layers.Dense(8, activation=tf.nn.relu)(input_users_2)
  x2_item = keras.layers.Dense(8, activation=tf.nn.relu)(input_items_2)
  
  #user graph + user bert
  concatenated_1 = keras.layers.Concatenate()([x1_user, x2_user])

  #item graph + item bert
  concatenated_2 = keras.layers.Concatenate()([x1_item, x2_item])

  #dense after concat
  dense_user = keras.layers.Dense(8, activation=tf.nn.relu)(concatenated_1)
  dense_item = keras.layers.Dense(8, activation=tf.nn.relu)(concatenated_2)

  concatenated = keras.layers.Concatenate()([dense_user, dense_item])

  #concatenated = keras.layers.Flatten()(concatenated)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(concatenated)
  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)
  return model

#CONFIGURAZIONE 2
def run_conf_2(X,y,dim_embeddings,epochs,batch_size):
  model = keras.Sequential()
  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))
  x1_user = keras.layers.Dense(8, activation=tf.nn.relu)(input_users_1)
  x1_item = keras.layers.Dense(8, activation=tf.nn.relu)(input_items_1)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))
  x2_user = keras.layers.Dense(8, activation=tf.nn.relu)(input_users_2)
  x2_item = keras.layers.Dense(8, activation=tf.nn.relu)(input_items_2)

  #user graph + item graph
  concatenated_1 = keras.layers.Concatenate()([x1_user, x1_item])

  #user bert + item bert
  concatenated_2 = keras.layers.Concatenate()([x2_user, x2_item])

  #dense after concat
  dense_graph = keras.layers.Dense(8, activation=tf.nn.relu)(concatenated_1)
  dense_bert = keras.layers.Dense(8, activation=tf.nn.relu)(concatenated_2)

  concatenated = keras.layers.Concatenate()([dense_graph, dense_bert])
  #concatenated = keras.layers.Flatten()(concatenated)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(concatenated)
  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)
  return model