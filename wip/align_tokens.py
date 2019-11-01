import numpy as np
from spacy.gold import align
from collections import defaultdict

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

if __name__=="__main__":
    bert_tokens=['[CLS]', 'Sy', '##mpi', '##esis', 'kara', '##gios', '##is', 'is', 'een', 'vliesvleugelig', 'insect', 'uit', 'de', 'familie', 'Eulophidae', '.', '[SEP]'] 
    our_tokens=['Sympiesis', 'karagiosis', 'is', 'een', 'vliesvleugelig', 'insect', 'uit', 'de', 'familie', 'Eulophidae', '.']
    #bert_tokens=['Sy', '##mpi', '##esis']
    #our_tokens=['Sympiesis']

    #bert_tokens=['[CLS]', 'after', '5', 'months', 'and', '48', 'games', ',', 'the', 'match', 'was', 'abandoned', 'in', 'controversial', 'circumstances', 'with', 'ka', '##rp', '##ov', 'leading', 'five', 'wins', 'to', 'three', '(', 'with', '40', 'draws', ')', ',', 'and', 'replay', '##ed', 'in', 'the', 'world', 'chess', 'championship', '1985', '.', '[SEP]']
    #spacy_tokens=['after', '5', 'months', 'and', '48', 'games', ',', 'the', 'match', 'was', 'abandoned', 'in', 'controversial', 'circumstances', 'with', 'karpov', 'leading', 'five', 'wins', 'to', 'three', '(', 'with', '40', 'draws', ')', ',', 'and', 'replayed', 'in', 'the', 'world', 'chess', 'championship', '1985', '.']
    results=align_bert_to_spacy(bert_tokens, our_tokens)
    print(results)
