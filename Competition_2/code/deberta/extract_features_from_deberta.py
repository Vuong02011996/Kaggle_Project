import torch
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DebertaV2Tokenizer, DebertaV2Model


"""
Deberta: Function to extract text features¶
("tx" in variable / function names refers to "transformer")
"""
# Load DeBERTa tokenizer and model
"""
Tokenizer: 
    + Splits the text into tokens (e.g., words, subwords, or characters) based on the model's vocabulary
    + Converts raw text to token IDs.
    + Output: Tokenized and encoded input (numerical IDs).
Model: 
    + Processes token IDs to produce outputs (e.g., embeddings, predictions).
    + Output: Embeddings, hidden states, or predictions.
        + Forward Pass: Passes the tokenized inputs through the model layers to produce outputs such as hidden states, logits, or embeddings.
        + Feature Extraction: Extracts features from the hidden states which can be used for downstream tasks.
        + Inference: Produces predictions or embeddings from the input data.
"""
tokenizer = DebertaV2Tokenizer.from_pretrained('/home/oryza/Desktop/KK/Competition_1/models/deberta-v3-pytorch-large-v1')

# .cuda() moves the model to the GPU for faster computation.
tx_model = DebertaV2Model.from_pretrained('/home/oryza/Desktop/KK/Competition_1/models/deberta-v3-pytorch-large-v1').cuda()


def batch_extract_tx_features(texts, tokenizer, model, batch_size=3, max_length=1440):
    total_texts = 0
    total_over_max_length = 0

    features = []
    model.eval()  # Set model to evaluation mode
    # Use autocast for mixed precision
    with torch.cuda.amp.autocast():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', max_length=max_length, truncation=True,
                               padding=True).to('cuda')

            input_ids = inputs['input_ids']
            # Check for truncation
            for j, input_id in enumerate(input_ids):
                total_texts += 1
                if (input_id == tokenizer.pad_token_id).sum() == 0 and input_id.shape[0] == max_length:
                    total_over_max_length += 1

            with torch.no_grad():
                outputs = model(**inputs)
            batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.extend(batch_features)
            # clear cache / print status "."
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                # print(".", end="")

    print("total_over_max_length: ", total_over_max_length)
    print("total_texts: ", total_texts)
    print("Ratio of texts over max_length tokens:", total_over_max_length / total_texts)
    return np.vstack(features)


"""
Deberta: Extract features for prompt and both responses
Also adding difference between two responses to array (seemed to help)
"""


def get_tx_vectors(df, columns_to_vectorize):
    vectors = []
    for column in tqdm(columns_to_vectorize, desc="Vectorizing Columns"):
        print("Vectorizing", column)
        vectors.append(batch_extract_tx_features(df[column].tolist(), tokenizer, tx_model))

    vectors = np.array(vectors)
    vectors = np.transpose(vectors, (1, 0, 2))

    # Compute average difference
    avg_dif = vectors[:, 1, :] - vectors[:, 2, :]
    avg_dif = avg_dif.reshape(vectors.shape[0], 1, vectors.shape[2])
    vectors = np.concatenate((vectors, avg_dif), axis=1)

    # Calculate cosine similarities and append them
    similarities = []
    for i in range(vectors.shape[0]):
        prompt_vec = vectors[i, 0, :].reshape(1, -1)
        response1_vec = vectors[i, 1, :].reshape(1, -1)
        response2_vec = vectors[i, 2, :].reshape(1, -1)

        # Cosine similarity between prompt and response1
        sim_prompt_resp1 = cosine_similarity(prompt_vec, response1_vec)[0][0]

        # Cosine similarity between prompt and response2
        sim_prompt_resp2 = cosine_similarity(prompt_vec, response2_vec)[0][0]

        # Cosine similarity between response1 and response2
        sim_resp1_resp2 = cosine_similarity(response1_vec, response2_vec)[0][0]

        similarities.append([sim_prompt_resp1, sim_prompt_resp2, sim_resp1_resp2])

    similarities = np.array(similarities)

    # Reshape vectors to 2D
    vectors = vectors.reshape(len(vectors), -1)

    # Concatenate vectors and similarities
    final_vectors = np.concatenate((vectors, similarities), axis=1)

    return final_vectors


