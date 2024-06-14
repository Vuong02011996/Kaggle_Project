# Neither PyTorch nor TensorFlow >= 2.0 have been found.Models won't be available and only tokenizers, configuration and file/data utilities can be used
+ https://stackoverflow.com/questions/64337550/neither-pytorch-nor-tensorflow-2-0-have-been-found-models-wont-be-available
+ pip install torch

# HuggingFace AutoTokenizer | ValueError: Couldn't instantiate the backend tokenizer
+ https://stackoverflow.com/questions/70698407/huggingface-autotokenizer-valueerror-couldnt-instantiate-the-backend-tokeniz
+ pip install sentencepiece

# AttributeError: module 'lib' has no attribute 'OpenSSL_add_all_algorithms'
+ pip install cryptography==38.0.4

# ValueError: FP16 Mixed precision training with AMP or APEX (`--fp16`) and FP16 half precision evaluation (`--fp16_full_eval`) can only be used on CUDA or MLU devices or NPU devices or certain XPU devices (with IPEX).
+ Because: print(torch.cuda.is_available()) is False
+ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
+ https://github.com/huggingface/transformers/issues/2704

# DebertaV2Converter requires the protobuf library but it was not found in your environment
+ protobuf==3.20.1
+ https://github.com/huggingface/transformers/issues/10020

# Show all dataframe in polars
+ https://stackoverflow.com/questions/75929721/how-to-show-full-column-width-of-polars-dataframe-in-python


# Python scikit svm "Vocabulary not fitted or provided"
+ https://stackoverflow.com/questions/60472925/python-scikit-svm-vocabulary-not-fitted-or-provided
+ NO CHANGE: transform(infer) -> fit_transform(train)
+ Cause error:
## Error when inference with CountVectorizer
  + ValueError: Number of features of the model must match the input. Model n_features_ is 2219 and input n_features is 1844
    + train: 2219 words <=> num of features
    + test: 1844 words <=> num of features
    `proba= model.predict(test_feats[feature_names])+ a`
  + FIX: Save vectorizer_cnt to pickle and load when infer:
    ```
       with open(path_vectorizer_models + 'vectorizer_cnt.pkl', 'wb') as f:
            pickle.dump(vectorizer_cnt, f)
    ```

# ModuleNotFoundError: No module named 'Competition_1' when infer
+ Only save file pickle and have f before
    ```
     Save the fitted vectorizer
        with open(f'vectorizer_cnt.pkl', 'wb') as f:
            pickle.dump(vectorizer_cnt, f)
    ```
# pickle.PicklingError: Can't pickle <function <lambda> at 0x7f9a7d971da0>: attribute lookup <lambda> 
+ change `tokenizer=lambda x: x,`
+ to
    ```commandline
    def identity_tokenizer(x):
        return x
    ```