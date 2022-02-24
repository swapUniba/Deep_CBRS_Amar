# Deep_CBRS_Amar
Amar deep architectures for content based recommendations

The following steps have to be performed on a Google Colab notebook.

**1.	Create a new notebook on Google Colab and insert as first code cell:**

      !git clone https://github.com/swapUniba/Deep_CBRS_Amar.git
      
The cell execution will clone the content of the repository to the virtual Google Colab storage. DBbook and Movielens training and test files are also provided in the folder “datasets”.

**2.	Insert as second cell:**

      cd Deep_CBRS_Amar

This cell execution is necessary to move into the project folder.

**3.	Add the necessary embeddings in the embeddings folder, in the following path:** ../Deep_CBRS_Amar/embeddings

**4.	Insert as third cell in the notebook one of these commands:**

      !python train_model1_graph.py                   (1st AMAR variant with OpenKE emb.)
      !python train_model1_bert.py                    (1st AMAR variant with BERT emb.)
      !python train_model1_graph_bert.py              (1st AMAR variant with BERT and OpenKE emb.)
      !python train_model2_conf.py                    (2nd AMAR variant with 2 configurations)
      !python train_model3_conf_att.py               (3rd AMAR variant with attention layer)
      
      !python train_model1_strategy.py                (1st AMAR variant with ELMO and smaller OpenKE emb.)
      !python train_model2_conf_strategy.py          (2nd AMAR variant with ELMO and smaller OpenKE emb)
      !python train_model3_conf_strategy_att.py      (3rd AMAR variant with attention layers with ELMO and smaller OpenKE emb)
      
      !python train_model1_graph_bert.py              (1st AMAR variant with USE and OpenKE emb.)
      !python train_model2_conf.py                    (2nd AMAR variant with USE and OpenKE emb)
      !python train_model3_conf_att.py               (3rd AMAR variant with attention layers with USE and OpenKE emb)
      
Based on the specific architecture to evaluate and the input embedding to use, it is necessary to choose the train files corresponding to the chosen architecture. **Before starting the execution, it is necessary to specify the embedding filenames in the chosen train file**


For 2nd and 3rd Amar variants, 2 input configurations are available:
- CONF1, Entity Based: (Graph items + Contextual items) + (Graph users + Contextual users)
- CONF2, Feature Based: (Graph items + Graph users) + (Contextual items + Contextual users)

The preferred confiuguration can be chosen in the train file.

After the execution of the training phase, the model will be created and stored in folder “results”.

**5.	Insert as fourth cell in the notebook one of these commands (should correspond to the train):**

      !python test_model1_graph.py
      !python test_model1_bert.py
      !python test_model1_graph_bert.py
      !python test_model2_conf.py
      !python test_model2_conf2_strategy.py
      !python test_model3_conf2_att.py
      !python test_model3_conf2_strategy_att.py
      
      !python test_model1_strategy.py
      !python test_model2_conf2_strategy.py
      !python test_model3_conf2_strategy_att.py
      
      !python test_model1_graph_bert.py
      !python test_model2_conf.py
      !python test_model3_conf2_att.py
      
Based on the specific architecture to evaluate and the input embedding to use, it is necessary to choose the test files corresponding to the chosen architecture. **Also in this case, before starting the execution, it is necessary to specify the embedding filenames in the chosen test file**. The testing phase will produce the top 5 and top 10 recommendations, stored in the folder “predictions”.

**6.	Insert as fifth cell in the notebook:**

      !python evaluate_results.py datasets/test2id_all_pred.txt predictions

The cell execution will compute three metrics: precision, recall and f1 score both for the top 5 and the top 10 predictions computed after the testing step and stored in the folder “predictions”. 

##Deep CBRS Amar Revisited
Some alternative models has been developed for BASIC, MIXED and EXTENDED architecures. You can find those new models on _models-alt_ directory. Thus, change the training and testing files accordingly, importing the desired models. Those new models maintain the same interface. 