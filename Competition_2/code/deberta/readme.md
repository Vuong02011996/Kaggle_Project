# Max_length
+ 1440: tx_train_vectors.shape (57477, 3075)
  + Vectorizing prompt
    + total_over_max_length:  572
    + total_texts:  57477
    + Ratio of texts over max_length tokens: 0.009951806809680394
  + Vectorizing response_a
    + total_over_max_length:  1101
    + total_texts:  57477
    + Ratio of texts over max_length tokens: 0.01915548828226943
  + Vectorizing response_b
    + total_over_max_length:  1143
    + Ratio of texts over max_length tokens: 0.0198862153557075
  + Result: 1.018
  + `5 fold: 1.015`
  
+ 2048: tx_train_vectors.shape (57477, 3075)
  + Vectorizing prompt
    + total_over_max_length:  359
    + total_texts:  57477
    + Ratio of texts over max_length tokens: 0.006245976651530177
  + Vectorizing response_a
    + total_over_max_length:  501
    + total_texts:  57477
    + Ratio of texts over max_length tokens: 0.008716530090296989
  + Vectorizing response_b
    + total_over_max_length:  511
    + Ratio of texts over max_length tokens: 0.008890512726829862
  + Result: 1.020
+ 3072: tx_train_vectors.shape (57477, 3075)
  + Vectorizing prompt
    + total_over_max_length:  141
    + total_texts:  57477
    + Ratio of texts over max_length tokens: 0.0024531551751135238
  + Vectorizing response_a
    + total_over_max_length:  171
    + total_texts:  57477
    + Ratio of texts over max_length tokens: 0.002975103084712146
  + Vectorizing response_b
    + total_over_max_length:  186
    + Ratio of texts over max_length tokens: 0.0032360770395114566
  + Result: 1.020
  + `5 fold: 1.015`
# Max_depth
+ 3
  + 1440: 1.019
+ 4
  + 1440: 1.018

# Inference

+ Train + infer: https://www.kaggle.com/code/richolson/deberta-tf-idf-word2vec-length
+ Models: https://www.kaggle.com/models/alexxxsem/deberta-v3/PyTorch/large
