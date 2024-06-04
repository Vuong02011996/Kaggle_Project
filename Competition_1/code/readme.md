# Deberta_v3_small
## Strategy 1 
+  I create a single tokenizer for all fold, tokenize the whole dataset first then split them into folds
+ use only xsmall deberta can get lb=0.802 with using linear, setting warm_up to 0.1 and change max length to 1440.
+ 