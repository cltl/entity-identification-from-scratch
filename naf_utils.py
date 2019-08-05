from lxml import etree
import glob
import datetime
import shutil

import spacy_to_naf

import config

def patch_classes_with_tokens(news_items, naf_dir, entity_layer):
    for item in news_items:
        docid=item.identifier
        naf_output_path = naf_dir / f'{docid}.naf'
        eid_to_tids=obtain_entity_data(naf_output_path, entity_layer)
        for e in item.sys_entity_mentions:
            eid=e.eid
            e.tokens=eid_to_tids[eid]
    return news_items

def load_sentences(naf_dir, iteration):
    all_sent=[]
    for f in glob.glob('%s/*.naf' % naf_dir):

        parser = etree.XMLParser(remove_blank_text=True)
        doc=etree.parse(f, parser)

        root=doc.getroot()
        s=load_sentences_from_naf(iteration, root)
        all_sent+=s
    return all_sent

# ------ NAF processing utils --------------------

def create_nafs(naf_folder, 
                news_items, 
                nl_nlp,
                layers={'raw', 'text', 'terms', 'entities'}):
    """Create NAFs if not there already."""
    
    if naf_folder.exists():
        file_count =  len(glob.glob('%s/*.naf' % naf_folder))
        assert file_count == len(news_items), 'NAF directory exists, but a wrong amount of files. Did you edit the source documents?'
        print('NAF files were already there, and at the right number.')
        #shutil.rmtree(str(naf_folder))
    else:
        print('No NAF files found. Let\'s create them.')
        naf_folder.mkdir()
        create_naf_for_documents(news_items, 
                                 layers, 
                                 nl_nlp, 
                                 naf_folder)

def create_naf_for_documents(news_items, layers, nlp, naf_folder, language='nl'):
    """Create NAF files for a collection of documents."""
   
    corpus_uri=config.corpus_uri
    for i, news_item in enumerate(news_items):
        text=f"{news_item.title}\n{news_item.content}"
        docid=news_item.identifier
        naf_output_path = naf_folder / f'{docid}.naf'
                
        
        process_spacy_and_convert_to_naf(nlp,
                                               text,
                                               language,
                                               uri=f'{corpus_uri}/{docid}',
                                               title=news_item.title,
                                               dct=datetime.datetime.now(),
                                               layers=layers,
                                               output_path=naf_output_path)
        

def process_spacy_and_convert_to_naf(nlp, 
                                     text, 
                                     language, 
                                     uri, 
                                     title, 
                                     dct, 
                                     layers, 
                                     output_path=None):
        """
        process with spacy and convert to NAF
        :param nlp: spacy language model
        :param datetime.datetime dct: document creation time
        :param set layers: layers to convert to NAF, e.g., {'raw', 'text', 'terms'}
        :param output_path: if provided, NAF is saved to that file
        :return: the root of the NAF XML object
        """
        root = spacy_to_naf.text_to_NAF(text=text,
                                        nlp=nlp,
                                        dct=dct,
                                        layers=layers,
                                        title=title,
                                        uri=uri,
                                        language=language)

        if output_path is not None:
            with open(output_path, 'w') as outfile:
                outfile.write(spacy_to_naf.NAF_to_string(NAF=root))

def add_ext_references_to_naf(all_docs, iter_id, in_naf_dir, out_naf_dir=None):

    if out_naf_dir is not None:
        if out_naf_dir.exists():
            shutil.rmtree(str(out_naf_dir))
        out_naf_dir.mkdir()
    
    entity_layer_str=config.naf_entity_layer
    for news_item in all_docs:
        docid=news_item.identifier
        infile = in_naf_dir / f'{docid}.naf'
        parser = etree.XMLParser(remove_blank_text=True)
        naf_file=etree.parse(infile, parser)
        
        root=naf_file.getroot()
        entities_layer=root.find(entity_layer_str)
        
        entities=news_item.sys_entity_mentions
        eid2identity={}
        for e in entities:
            
            eid2identity[e.eid]=e.identity
        for naf_entity in entities_layer.findall('entity'):
            eid=naf_entity.get('id')
            identity=eid2identity[eid]
            ext_refs=naf_entity.find('externalReferences')
            ext_ref=etree.SubElement(ext_refs, 'externalReference')
            ext_ref.set('target', identity)
            ext_ref.set('source', iter_id)

        if out_naf_dir is not None:
            outfile_path = out_naf_dir / f'{docid}.naf'
            with open(outfile_path, 'w') as outfile:
                    outfile.write(spacy_to_naf.NAF_to_string(NAF=root))

def load_sentences_from_naf(iteration, root):
    to_replace={}
    
    
    ent_layer=root.find(config.naf_entity_layer)
    for e in ent_layer.findall('entity'):
        # get identity
        ext_refs=e.find('externalReferences')
        the_id=''
        for er in ext_refs.findall('externalReference'):
            if er.get('source')=='iteration%d' % iteration:
                the_id=er.get('target')
                
        # get spans
        refs=e.find('references')
        span=refs.find('span')
        for target in span.findall('target'):
            t=target.get('id')
            to_replace[t]=''
        to_replace[t]=the_id
    
    token_layer=root.find('text')
    old_sent='1'
    sentences=[]
    current_sentence=[]
    for w in token_layer.findall('wf'):
        idx=w.get('id').replace('w', 't')
        sent=w.get('sent')
        txt=w.text
        if old_sent!=sent:
            sentences.append(current_sentence)
            current_sentence=[]
        if idx in to_replace:
            if to_replace[idx]:
                current_sentence.append(to_replace[idx])
        else:
            current_sentence.append(txt)
        old_sent=sent
    sentences.append(current_sentence)
    return sentences

def obtain_entity_data(naf_file, entities_layer_id):
    parser = etree.XMLParser(remove_blank_text=True)
    doc=etree.parse(naf_file, parser)

    root=doc.getroot()

    entities_layer=root.find(entities_layer_id)
    eid_to_tids={}
    for entity in entities_layer.findall('entity'):
        eid=entity.get('id')
        refs=entity.find('references')
        span=refs.find('span')
        tids=[]
        for target in span.findall('target'):
            tids.append(target.get('id'))
        eid_to_tids[eid]=tids
    return eid_to_tids