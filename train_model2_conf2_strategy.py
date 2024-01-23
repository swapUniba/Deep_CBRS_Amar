from utilities.utils import (
    read_ratings,
    read_graph_embeddings,
    read_bert_embedding,
    matching_Bert_Graph,
)
from models.model2_conf2_strategy import run_model

graph_embeddings = read_graph_embeddings("embeddings/HolEembedding_768.json")
user_bert_embeddings = read_bert_embedding(
    "embeddings/elmo_user_embeddings_nostopw_1024.json"
)
item_bert_embeddings = read_bert_embedding(
    "embeddings/elmo_embeddings_nostopw_1024.json"
)

user, item, rating = read_ratings("datasets/movielens/train2id.tsv")
X_graph, X_bert, dim_graph, dim_bert, y = matching_Bert_Graph(
    user, item, rating, graph_embeddings, user_bert_embeddings, item_bert_embeddings
)
model = run_model(X_graph, X_bert, dim_graph, dim_bert, y, epochs=25, batch_size=1536)


# creates a HDF5 file 'model.h5'
model.save("results/model.h5")
