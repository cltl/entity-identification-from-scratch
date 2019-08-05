import torch
from collections import defaultdict
import glob
from lxml import etree
from gensim.models import Word2Vec
import os.path

import load_utils
import naf_utils as naf

def get_bert_embeddings(tokens, model, tokenizer):
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
    
    sentence_embedding = torch.mean(encoded_layers[10], 1)
    
    return sentence_embedding[0]
    
def get_sentence_embeddings(naf_dir, iteration, model, tokenizer):
    sent_emb=defaultdict(dict)
    print(naf_dir)
    for f in glob.glob('%s/%d/*.naf' % (naf_dir, iteration-1)):
        parser = etree.XMLParser(remove_blank_text=True)
        doc=etree.parse(f, parser)

        root=doc.getroot()
        s=naf.load_sentences_from_naf(iteration, root)
        
        for index, sentence in enumerate(s):
            emb=get_bert_embeddings(sentence, model, tokenizer)
            sent_index=str(index+1)
            doc_id=(f.split('/')[-1]).split('.')[0]
            sent_emb[doc_id][sent_index]=emb
        print(doc_id)
        
    return sent_emb
    
def sent_to_id_embeddings(sent_embeddings, data):
    entity_embs=defaultdict(list)
    for news_item in data:
        doc_id = news_item.identifier
        print('loading')
        print(doc_id)
        for em in news_item.sys_entity_mentions:
            sentence=em.sentence
            sent_emb=sent_embeddings[doc_id][sentence]
            identity=em.identity
            entity_embs[identity].append(sent_emb)
            
    agg_entity_embs={}
    for identity, embs in entity_embs.items():
        emb_array=np.array(embs)
        agg_entity_embs[identity]=np.mean(emb_array, axis=0)
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