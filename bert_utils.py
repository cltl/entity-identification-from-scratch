import torch
from gensim.models import Word2Vec
import os.path
import numpy as np
import pickle
from collections import defaultdict
from spacy.gold import align
import spacy

import pickle_utils as pkl

#spacy.gold.USE_NEW_ALIGN = True

# ---------- Alignment functions ------- #
def remove_bertie_stuff(tokens):
    new_tokens=[]
    for i, token in enumerate(tokens):
        if i==0 or i==len(tokens)-1: continue
        if token.startswith('##'): token=token[2:]
        new_tokens.append(token)
    return new_tokens

def align_bert_to_spacy(bert_tokens, spacy_tokens):
    other_tokens=remove_bertie_stuff(bert_tokens)
#    print(other_tokens)
#    other_tokens=bert_tokens
    cost, b2s, s2b, b2s_multi, s2b_multi = align(other_tokens, spacy_tokens)
    print("Misaligned tokens:", cost)  # 2
    print("One-to-one mappings bert -> spacy", b2s)  # array([0, 1, 2, 3, -1, -1, 5, 6])
    print("One-to-one mappings spacy -> bert", s2b)  # array([0, 1, 2, 3, 5, 6, 7])
    print("Many-to-one mappings bert -> spacy", b2s_multi)  # {4: 4, 5: 4}
    print("Many-to-one mappings spacy -> bert", s2b_multi)  # {}

    bert2spacy=defaultdict(list)
    spacy2bert=defaultdict(list)
    for bert_index, spacy_index  in enumerate(b2s):
        print(bert_index, spacy_index)
        if spacy_index!=-1:
            bert2spacy[bert_index].append(spacy_index)
            spacy2bert[spacy_index].append(bert_index)
        elif bert_index in b2s_multi.keys():
            bert2spacy[bert_index].append(b2s_multi[bert_index])
            spacy2bert[b2s_multi[bert_index]].append(bert_index)
        else:
            bert2spacy[bert_index].append(-1)
            spacy2bert[-1].append(bert_index)
    return bert2spacy, spacy2bert

def map_bert_embeddings_to_tokens(bert_tokens,
                                    our_tokens,
                                    entities,
                                    bert_embeddings,
                                    sent_id,
                                    offset,
                                    verbose):
    print(bert_tokens, our_tokens)
    bert2our, our2bert=align_bert_to_spacy(bert_tokens, our_tokens)
    print(bert2our, our2bert)
    entity_embs={}
    for entity in entities:
        if entity.sentence != sent_id:
            continue
        entity_bert_tokens=[]
        for our_token in entity.tokens:
            numeric_id=int(our_token.strip('t')) - offset
            if numeric_id in our2bert.keys():
                bert_tokens=our2bert[numeric_id]
                entity_bert_tokens+=bert_tokens
            else:# we did not manage to map this
                print('UNMAPPED token id', our_token)
                pass

        embs = np.zeros(len(bert_embeddings[0]))
        for tid in entity_bert_tokens:
            embs += np.array(bert_embeddings[tid])
        entity_embs[entity.eid] = embs
    return entity_embs

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


if __name__=="__main__":
    bert_tokens=['[CLS]', 'Sy', '##mpi', '##esis', 'kara', '##gios', '##is', 'is', 'een', 'vliesvleugelig', 'insect', 'uit', 'de', 'familie', 'Eulophidae', '.', '[SEP]']
    our_tokens=['Sympiesis', 'karagiosis', 'is', 'een', 'vliesvleugelig', 'insect', 'uit', 'de', 'familie', 'Eulophidae', '.']
    #bert_tokens=['Sy', '##mpi', '##esis']
    #our_tokens=['Sympiesis']

    #bert_tokens=['[CLS]', 'after', '5', 'months', 'and', '48', 'games', ',', 'the', 'match', 'was', 'abandoned', 'in', 'controversial', 'circumstances', 'with', 'ka', '##rp', '##ov', 'leading', 'five', 'wins', 'to', 'three', '(', 'with', '40', 'draws', ')', ',', 'and', 'replay', '##ed', 'in', 'the', 'world', 'chess', 'championship', '1985', '.', '[SEP]']
    #spacy_tokens=['after', '5', 'months', 'and', '48', 'games', ',', 'the', 'match', 'was', 'abandoned', 'in', 'controversial', 'circumstances', 'with', 'karpov', 'leading', 'five', 'wins', 'to', 'three', '(', 'with', '40', 'draws', ')', ',', 'and', 'replayed', 'in', 'the', 'world', 'chess', 'championship', '1985', '.']
    results=align_bert_to_spacy(bert_tokens, our_tokens)
    print(results)
