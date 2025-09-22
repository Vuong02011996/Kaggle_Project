# GPU - model Gemma2-2B-it
+ https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945

+ This model has 9B parameters compared with Ettin-Encoder-1B which has 1B parameters. 
+ This model is difficult to train on most GPUs. For efficient training, we need to use LORA or QLORA
+ When training without QLoRA, we cannot do this in Kaggle's 2xT4 GPU notebook, we need to use more GPU offline. 
+ I have verified that the settings above will work on 8xV100 32GB GPU or 4xA100 80GB GPU

Yes. When training offline, we use bf16=True and fp16=False in Trainer arguments. 
This notebook is inference and has bf16=False and fp16=True (to be compatible with Kaggle's T4 GPU).

We cannot use bf16=True on Kaggle's T4 GPU because it is too old. So if you want to train on Kaggle's T4, then you need to use fp32 (i.e. bf16=False and fp16=False).