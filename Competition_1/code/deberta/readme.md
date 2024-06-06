# Model Deberta

# Deberta_v3_small

+ train_batch_size = 4
+ eval_batch_size = 8
+ train_epochs = 4
+ train max_length 1440
+ infer max_length 1536
+ warm_up to 0.1
+ n_splits = 5

## Strategy 1 - version 2
+  I create a single tokenizer for all fold, tokenize the whole dataset first then split them into folds
+ use only xsmall deberta can get lb=0.802 with using linear, setting **warm_up to 0.1** and change max length to 1440.
+ Train max_length 1024, infer max_length 1536
+ train_batch_size = 4
+ eval_batch_size = 8
+ **train_epochs = 4**
+ train max_length 1440
+ infer max_length 1536
+ warm_up to 0.1
+ Result
  + Overall QWK CV = Delete file csv validation??
  + LB: 0.804

## Strategy 2 - version 3
+ **train_epochs = 15**
+ Result
  + Overall QWK CV = 0.8175799710266693
  + LB: 0.804
  
## Strategy 3 - version 4
+ **USES CLASSIFICATION**
+ Result:
  + Overall QWK CV = 0.8078820585361615


## Strategy 4 - version 5
+ **add on persuade_2.0 DATA**
+ Result
  + Overall QWK CV = 
  + LB: 

## Strategy 5 - version 6
+ **add on persuade_2.0 DATA**
+ n_splits = 15
+ Result
  + Overall QWK CV = 
  + LB: 

