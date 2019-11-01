import torch
from gensim.models import Word2Vec
import os.path
import numpy as np
import pickle
from collections import defaultdict

import pickle_utils as pkl



# ---------- Helper functions ---------- #

def get_bert_sentence_embeddings(encoded_layers):
    """Obtain sentence embeddings by averaging all embeddings in the second last layer for a sentence."""
    sent_emb = torch.mean(encoded_layers[-2], 1)
    return sent_emb[0]


def get_token_embeddings(tokenized_text, encoded_layers):
    """
    Convert the hidden state embeddings into single token vectors.
    """

    # Holds the list of 12 layer embeddings for each token
    # Will have the shape: [# tokens, # layers, # features]
    token_embeddings = []

    batch_i = 0

    # For each token in the sentence...
    for token_i in range(len(tokenized_text)):
        # Holds 12 layers of hidden states for each token 
        hidden_layers = []

        # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):
            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][batch_i][token_i]

            hidden_layers.append(vec)

        token_embeddings.append(hidden_layers)
    return token_embeddings


def get_bert_word_embeddings(tokenized_text, encoded_layers):
    """
    Get BERT word embeddings by concatenating the last 4 layers for a token.
    """
    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []

    token_embeddings = get_token_embeddings(tokenized_text, encoded_layers)

    # For each token in the sentence...
    for token in token_embeddings:
        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), 0)

        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)
    return token_vecs_cat


def get_bert_embeddings(text, model, tokenizer):
    """
    Obtain BERT embeddings for a text.
    """
    print(text)
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    print(tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    return tokenized_text, encoded_layers
