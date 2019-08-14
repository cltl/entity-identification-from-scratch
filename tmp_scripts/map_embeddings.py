from collections import defaultdict

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

def map_bert_embeddings_to_tokens(berts, entities):
    norm_bert, mapping_old_new_bert = get_bert_mappings(berts)

    for ek, ev in entities.items():
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

        print(ek, ev, closest_diff, closest_tids)


berts=['[CLS]', 'Leipzig', 'is', 'een', 'kr', '##eis', '##fre', '##ie', 'Stadt', 'in', 'Duitsland', 'gelegen', 'aan', 'de', 'Pl', '##ei', '##ße', 'met', '521', '.', '000', 'inwoners', '(', '2012', ')', '.', '[SEP]']

entities={tuple([0]): ['Leipzig'], tuple([3,4]): ['kreisfreie', 'Stadt'], tuple([6]): ['Duitsland'], tuple([10]): ['Pleiße'], tuple([15]): ['2012']}

map_bert_embeddings_to_tokens(berts, entities)

