import torch
from collections import defaultdict
import glob
from lxml import etree
from gensim.models import Word2Vec
import os.path
import numpy as np
import pickle
import sys
from collections import defaultdict

import load_utils
import naf_utils as naf

def get_bert_mappings(berts):
    new_i=0
    new_bert=[]
    mapping=defaultdict(list)
    for bert_i, bert_token in enumerate(berts):
        if bert_i==0 or bert_i==len(berts)-1: continue
        if bert_i>1 and not bert_token.startswith('##'):
            print(new_i, current_token)
            new_bert.append(current_token)
            current_token=bert_token
            mapping[new_i].append(bert_i)
            new_i+=1
        elif bert_i==1:
            current_token=bert_token
        else:
            current_token+=bert_token[2:]
            mapping[new_i].append(bert_i)
    print(new_i, current_token)
    new_bert.append(current_token)

    print(berts, new_bert, mapping)
    return new_bert, mapping

def get_embedding_tids(tids, mapping):
    mapped=[]
    for t in tids:
        mapped+=mapping[t]
    return mapped

def map_bert_embeddings_to_tokens(berts, entities, word_embeddings, sent_id):
    norm_bert, mapping_old_new_bert = get_bert_mappings(berts)

    entity_embs={}

    for entity in entities:
        if entity.sentence!=sent_id: continue
        ev=entity.mention.split()
        ek=list(range(entity.begin_index, entity.end_index+1))
        closest_diff=999
        closest_tids=[]
        for bert_i, berts_token in enumerate(norm_bert):
            if berts_token==ev[0]:
                if len(ev)==1:
                    diff=abs(bert_i-ek[0])
                    if diff<closest_diff:
                        closest_diff=diff
                        raw_tids=[bert_i]
                        closest_tids=get_embedding_tids(raw_tids, mapping_old_new_bert)
                else:
                    fits=True
                    raw_tids=[]
                    for i, t in enumerate(ev):
                        if t!=norm_bert[bert_i+i]:
                            fits=False
                            break
                        else:
                            raw_tids.append(bert_i+i)
                    if fits:
                        diff=abs(bert_i-ek[0])
                        if diff<closest_diff:
                            closest_diff=diff
                            closest_tids=get_embedding_tids(raw_tids, mapping_old_new_bert)
        embs=np.zeros(len(word_embeddings[0]))
        if len(closest_tids)==1:
            entity_embs[entity.eid]=np.array(word_embeddings[closest_tids[0]])
        else:
            for tid in closest_tids:
                embs+=np.array(word_embeddings[tid])
            entity_embs[entity.eid]=embs
    return entity_embs

# -------- BERT Mapping functions ready ------ #

def get_bert_sentence_embeddings(encoded_layers):
    sent_emb=torch.mean(encoded_layers[-2], 1)
    return sent_emb[0]

def get_token_embeddings(tokenized_text, encoded_layers):
    """
    Convert the hidden state embeddings into single token vectors.
    """

    # Holds the list of 12 layer embeddings for each token
    # Will have the shape: [# tokens, # layers, # features]
    token_embeddings = [] 

    batch_i=0

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

    token_embeddings=get_token_embeddings(tokenized_text, encoded_layers)

    # For each token in the sentence...
    for token in token_embeddings:
        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), 0)
        
        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)
    return token_vecs_cat

def get_bert_embeddings(tokens, model, tokenizer):
    """
    Obtain BERT embeddings
    """
    text=' '.join(tokens)
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)

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
       
def get_word_and_sentence_embeddings(naf_dir, iteration, model, tokenizer, news_items, modify_entities=False):
    """
    Obtain word and sentence embeddings using BERT for an entire NAF collection.
    """
    sent_emb=defaultdict(dict)
    ent_emb=defaultdict(dict)

    # load NAF files from the previous iteration
    for f in glob.glob('%s/%d/*.naf' % (naf_dir, iteration-1)):
        parser = etree.XMLParser(remove_blank_text=True)
        doc=etree.parse(f, parser)

        root=doc.getroot()
        
        # load the sentences from this NAF file
        s=naf.load_sentences_from_naf(iteration, 
                                    root, 
                                    modify_entities=modify_entities)

        doc_id=(f.split('/')[-1]).split('.')[0]

        # Retrieve entities (TODO: make this more efficient, and from NAF)
        for item in news_items:
            if doc_id!=item.identifier: continue
            entities=item.sys_entity_mentions

        for index, sentence in enumerate(s):
            sent_index=str(index+1)

            tokenized_text, encoded_layers=get_bert_embeddings(sentence, model, tokenizer)

            # Get sentence embeddings
            sentence_embeddings = get_bert_sentence_embeddings(encoded_layers)
            print('Sentence embedding shape', sentence_embeddings.shape[0])
            sent_emb[doc_id][sent_index]=sentence_embeddings

            # Concatenated word embeddings
            word_embeddings=get_bert_word_embeddings(tokenized_text, encoded_layers)
            print('Word embeddings shape', len(word_embeddings), len(word_embeddings[0]))
            
            entity_embeddings=map_bert_embeddings_to_tokens(tokenized_text, entities, word_embeddings, sent_index)
            for eids, embs in entity_embeddings.items():
                ent_emb[doc_id][eids]=embs
            sys.exit()
        
    return ent_emb, sent_emb
    
def sent_to_id_embeddings(sent_embeddings, data):
    entity_embs=defaultdict(list)
    for news_item in data:
        doc_id = news_item.identifier
        for em in news_item.sys_entity_mentions:
            sentence=str(em.sentence)
            sent_emb=sent_embeddings[doc_id][sentence]
            identity=em.identity
            entity_embs[identity].append(sent_emb)
            
    agg_entity_embs={}
    for identity, embs in entity_embs.items():
        emb_arrays=[]
        if len(embs)>1:
            for e in embs:
                emb_arrays.append(np.array(e))
            agg_entity_embs[identity]=np.mean(np.array(emb_arrays), axis=0)
            #agg_entity_embs[identity]=np.array(embs[0]) #np.mean(np.array(emb_arrays), axis=0)
        else:
            agg_entity_embs[identity]=np.array(embs[0])
    with open('debug/bert_embs.p', 'wb') as wf:
        pickle.dump(agg_entity_embs, wf)
    return agg_entity_embs
    
def generate_w2v_embeddings(all_sentences,
                        save_loc='', 
                        pretrained=None):
    """Generate embeddings based on a collection of news items with entity identity, potentially starting from a pre-trained model."""
    if save_loc and os.path.isfile(save_loc):
        model = Word2Vec.load(save_loc)
        return model
    if pretrained:
        model_2 = Word2Vec(size=200, 
                            min_count=1,
                            workers=2,     # Number of processors (parallelisation)
                            window=5,      # Context window for words during training
                            iter=30)
        model_2.build_vocab(all_sentences)
        total_examples = model_2.corpus_count
        model = KeyedVectors.load_word2vec_format(pretrained, binary=False)
        model_2.build_vocab([list(model.vocab.keys())], update=True)
        model_2.intersect_word2vec_format(pretrained, binary=False, lockf=1.0)
        model_2.train(all_sentences, total_examples=total_examples, epochs=model_2.iter)
    else:
        
        model_2 = Word2Vec(all_sentences,
                         min_count=1,   # Ignore words that appear less than this
                         size=200,      # Dimensionality of word embeddings
                         workers=2,     # Number of processors (parallelisation)
                         window=5,      # Context window for words during training
                         iter=30)       # Number of epochs training over corpus
    if save_loc:
        model_2.save(save_loc)
    return model_2

def embeddings_in_a_doc(embeddings, d):
    for e in embeddings:
        if d in e:
            print(e)

def identity_vs_embeddings_stats(data, embeddings):
    stats=defaultdict(int)
    for news_item in data:
        for entity_mention in news_item.sys_entity_mentions:
            identity=load_utils.strip_identity(entity_mention.identity)
            #print(entity_mention.mention, entity_mention.identity, identity, identity in embeddings)
            stats[identity in embeddings]+=1
            if not identity in embeddings:
                print(identity)
    print(stats)
