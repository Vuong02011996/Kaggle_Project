# LGBM 
## Version 1 - 42s
+ Sentence feature: Features Number after Sentence preprocess:  28 - 8s
+ Word feature: Features Number after Word preprocess:  49 - 10s
+ LB: 0.757

## Version 2 - 48s
+ Sentence feature: 28
+ Word feature: 49
+ CountVectorizer: Features number after CountVectorizer:  235 - 14s
  + `token_pattern=r'\b\w+\b',`
  + ngram_range=(2, 3),
+ LB: 0.776

## Version 3 - 53s
+ Sentence feature: 28
+ Word feature: 49
  + `token_pattern=r'\b\w+\b',`
  + ngram_range=(3, 6),
    + Features number after CountVectorizer:  57 - 24s
    + (17307, 59)
+ LB: 0.761

## Version 6 - train + infer: 6529.4s - GPU P100
+ paragraph - Number of Features:  53 - 1294s
+ Features Number after Sentence preprocess:  81 - 14s
+ Features Number after Word preprocess:  102 - 20s
+ TfidfVectorizer - Number of Features:  19729 - 206s - lamda
+ Features number after CountVectorizer:  21899 - 75s - lamda
+ LB : 0.801

## Version 7 
+ paragraph - Number of Features:  53 - 558s
+ Features Number after Sentence preprocess:  81 - 7s
+ Features Number after Word preprocess:  102 - 8s
+ TfidfVectorizer - Number of Features:  765 - 9s - token_pattern=r'\b\w+\b'
  + ngram_range=(2,3),
  + min_df=0.05,
  + max_df=0.95,
+ CountVectorizer - Number of Features:  951 - 9s - token_pattern=r'\b\w+\b'
  + ngram_range=(2,3),
  + min_df=0.05,
  + max_df=0.95,
  
+ (17307, 953)
+ LB : 0.800

## Version 8
+ 
