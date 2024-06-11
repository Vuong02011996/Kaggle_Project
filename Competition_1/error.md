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
+ 