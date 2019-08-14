import glob
import sklearn.metrics
from lxml import etree

import config

def convert_str_to_int_list(str_list):
    unique_values = list(set(str_list))
    print('Num of clusters', len(unique_values))
    clusters=[]
    for element in str_list:
        clusters.append(unique_values.index(element))
    return clusters

def compute_rand_score(g, s):
    return sklearn.metrics.adjusted_rand_score(g, s)

def evaluate_naf_collection(naf_dir, iteration):
    sys_source=f'iteration{iteration}'
    entity_layer_str=config.naf_entity_layer
    gold_links=[]
    sys_links=[]
    for f in glob.glob('%s/*.naf' % naf_dir):
        parser = etree.XMLParser(remove_blank_text=True)
        doc=etree.parse(f, parser)

        root=doc.getroot()
        entities_layer=root.find(entity_layer_str)

        for naf_entity in entities_layer.findall('entity'):
            ext_refs=naf_entity.find('externalReferences')
            for ext_ref in ext_refs.findall('externalRef'):
                source=ext_ref.get('source')
                reference=ext_ref.get('reference')
                if source==sys_source:
                    sys_links.append(reference)
                elif not source:
                    gold_links.append(reference)

    print('GOLD')
    gold_clusters=convert_str_to_int_list(gold_links)

    print('SYS')
    sys_clusters=convert_str_to_int_list(sys_links)

    for gl, gc, sl, sc in zip(gold_links, gold_clusters, sys_links, sys_clusters):
        print('%s\t%d\t%s\t%d' % (gl, gc, sl, sc))

    rand_score=compute_rand_score(gold_clusters, sys_clusters)

    return rand_score

#l=['dog', 'cat', 'dog', 'walrus']
#clusters=convert_str_to_int_list(l)
#print(clusters)

iteration=1
for s in glob.glob('%s/*' % config.sys_dir):
    naf_dir='%s/naf' % s
    evaluation_score=evaluate_naf_collection(f'{naf_dir}/{iteration}', iteration)
    print(evaluation_score)
