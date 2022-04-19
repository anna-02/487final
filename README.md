# 487final
# Sarcasm Detection on Reddit using Hierarchal Attention Network with Context and Reply

This is a final course project for EECS 487 at the University of Michigan, by Anna Ablove (<aablove@umich.edu>) and Dan Choe (<danchoe@umich.edu>).

**References**

This project uses https://github.com/uvipen/Hierarchical-attention-networks-pytorch by Viet Nyugen for the hierarchal attention network architecture.

**Data**

This project uses the SARC dataset from A Large Self-Annotated Corpus for Sarcasm and 50 dimensional glove word vectors from https://www.kaggle.com/datasets/watts2/glove6b50dtxt.

"487final/sarcasm-detection-model" includes processing_basic.ipynb. 
processing_basic.ipynb generates our training/validation/test splits.

**Running the Model**

In "487final/sarcasm-detection-model", run_model.py trains the hierarchical attention with context and reply using train.csv, selects the best hyperparameters using dev.csv, and tests the model using test.csv.

train_a.py includes our training, parameter selection, and testing functions.

"487final/sarcasm-detection-model/src" includes the individual models, word-level attention, sentence-level attention model, hierarchical attention, and the final, which concatenates the vector representations of the context and reply and passes them through a feed-forward linear layer, followed by a softmax.

**Results**

The generated loss plots are stored in the "figures" directory. It includes the plots for our models using training sizes of 10k and 25k.

Our results, which include the precision, recall, F1 scores for each label (Sarcastic and Non-Sarcastic) as well as the accuracy get outputted to the "sarcasm-detection-model/outputs" directory. We include the results of our models for each training size we tested (3.4k, 4.4k, 6k, 10k, 25k).


