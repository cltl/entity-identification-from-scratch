from lxml import etree
import glob
import datetime
import shutil
import pathlib
from deprecated import deprecated

import spacy_to_naf

import classes
import config


# ------ Loading certain data from NAF files --------------------

def patch_classes_with_entities(news_items, naf_dir, entity_layer):
    """Obtain entity data from NAF and enrich the corresponding class objects."""
    for item in news_items:
        docid = item.identifier
        naf_output_path = naf_dir / f'{docid}.naf'
        ent_mention_objs=obtain_entity_data(naf_output_path, entity_layer)
        item.sys_entity_mentions=ent_mention_objs

    return news_items

def load_sentences_from_naf(iteration, root, naf_entity_layer, modify_entities):
    """Load sentences from a single NAF file (already loaded). Potentially replace entity mentions with their identity."""

    if modify_entities:
        to_replace=map_mentions_to_identity(root, naf_entity_layer)

    # Create list of lists of sentences in a file
    token_layer = root.find('text')
    old_sent = '1'
    sentences = []
    current_sentence = []
    for w in token_layer.findall('wf'):
        idx = w.get('id').replace('w', 't')
        sent = w.get('sent')
        txt = w.text
        if old_sent != sent:
            sentences.append(current_sentence)
            current_sentence = []
        if not modify_entities or idx not in to_replace:
            current_sentence.append(txt)
        elif idx in to_replace and to_replace[idx]:
            current_sentence.append(to_replace[idx])
        old_sent = sent
    sentences.append(current_sentence)
    return sentences

def load_sentences(naf_dir, iteration, modify_entities=False):
    """Load sentences from NAF files into a list of lists."""
    all_sent = []
    for f in glob.glob('%s/*.naf' % naf_dir):
        parser = etree.XMLParser(remove_blank_text=True)
        doc = etree.parse(f, parser)

        root = doc.getroot()
        s = load_sentences_from_naf(iteration, root, modify_entities)
        all_sent += s
    print(all_sent)
    return all_sent

def map_mentions_to_identity(root, naf_entity_layer):
    """Create a mapping between entity IDs and their identity."""
    to_replace = {}

    ent_layer = root.find(naf_entity_layer)
    for e in ent_layer.findall('entity'):
        # get identity
        ext_refs = e.find('externalReferences')
        the_id = ''
        for er in ext_refs.findall('externalRef'):
            if er.get('source') == 'iteration%d' % iteration:
                the_id = er.get('reference')

        # get spans
        refs = e.find('references')
        span = refs.find('span')
        for target in span.findall('target'):
            t = target.get('id')
            if target != span.findall('target')[-1]:
                to_replace[t] = ''
            else:
                to_replace[t] = the_id
    return to_replace

def obtain_entity_data(naf_file, entities_layer_id):
    """Obtain entity data from a NAF file."""
    entity_mentions=[]

    parser = etree.XMLParser(remove_blank_text=True)
    doc = etree.parse(naf_file, parser)

    root = doc.getroot()

    entities_layer = root.find(entities_layer_id)

    token_layer=root.find('text')
    wf2data={}
    for wf in token_layer.findall('wf'):
        wf_id=wf.get('id')
        wf2data[wf_id]={'sent': wf.get('sent'), 'offset': int(wf.get('offset')), 'length': int(wf.get('length')), 'text': wf.text}

    for entity in entities_layer.findall('entity'):
        eid = entity.get('id')
        refs = entity.find('references')
        span = refs.find('span')
        tids = []
        for target in span.findall('target'):
            tids.append(target.get('id'))
        
        wid=tids[0].replace('t', 'w')
        wdata=wf2data[wid]
        sent_id=int(wdata['sent'])
        begin_index=wdata['offset']
        last_wid=tids[-1].replace('t', 'w')
        end_index=wf2data[last_wid]['offset'] + wf2data[last_wid]['length']
        
        mention_list=[]
        for tid in tids:
            wid=tid.replace('t', 'w')
            mention_list.append(wf2data[wid]['text'])
        mention=' '.join(mention_list)

        ent_mention_obj=classes.EntityMention(
            eid=eid,
            the_type=entity.get('type'),
            tokens=tids,
            mention=mention,
            begin_index=begin_index,
            end_index=end_index,
            sentence=sent_id
        )
        entity_mentions.append(ent_mention_obj)
    return entity_mentions

# ------ NAF creation and editing utils --------------------

def create_nafs(naf_folder,
                news_items,
                nl_nlp,
                corpus_uri,
                ner_system='spacy',
                layers={'raw', 'text', 'terms', 'entities'}, recreate=True):
    """Create NAFs if not there already (includes SpaCy processing if no gold NER is given)."""

    print('NAF directory: ', naf_folder)
    if naf_folder.exists() and recreate:
        print('NAF directory exists. Removing and recreating...')
        shutil.rmtree(str(naf_folder))
    pathlib.Path(naf_folder).mkdir(parents=True, exist_ok=True)
    create_naf_for_documents(news_items,
                             layers,
                             nl_nlp,
                             naf_folder,
                             corpus_uri)


def create_naf_for_documents(news_items, layers, nlp, naf_folder, corpus_uri, language='nl'):
    """Create NAF files for a collection of documents."""

    for i, news_item in enumerate(news_items):
        text = f"{news_item.title}\n{news_item.content}"
        text = text.strip()
        docid = news_item.identifier
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

@deprecated(reason="The mentions are now first loaded in NAF and then in the python classes. This function is hence unused.")
def add_mentions_to_naf(all_docs, source_id, entity_layer_str, in_naf_dir, out_naf_dir=None):
    """Add entity mentions to a NAF file, based on gold mentions or based on spacy output."""
    if out_naf_dir is not None:
        if out_naf_dir.exists():
            shutil.rmtree(str(out_naf_dir))
        out_naf_dir.mkdir()

    for news_item in all_docs:
        docid = news_item.identifier
        infile = in_naf_dir / f'{docid}.naf'
        parser = etree.XMLParser(remove_blank_text=True)
        naf_file = etree.parse(infile, parser)

        root = naf_file.getroot()
        entities_layer = root.find(entity_layer_str)
        if entities_layer is None:
            entities_layer = etree.SubElement(root, entity_layer_str)

        if source_id == 'gold':
            entities = news_item.gold_entity_mentions
            for e in entities:
                entity_data = spacy_to_naf.EntityElement(e.eid,
                                                         e.the_type or '',
                                                         e.tokens,
                                                         e.mention,
                                                         [{'reference': e.identity, 'source': 'gold'}])
                spacy_to_naf.add_entity_element(entities_layer, entity_data)
        else:
            entities = news_item.sys_entity_mentions
            if not entities_layer.findall('entity'):
                for e in entities:
                    entity_data = spacy_to_naf.EntityElement(e.eid,
                                                         e.the_type or '',
                                                         e.tokens,
                                                         e.mention,
                                                         [])
                    spacy_to_naf.add_entity_element(entities_layer, entity_data)
        if out_naf_dir is not None:
            outfile_path = out_naf_dir / f'{docid}.naf'
            with open(outfile_path, 'w') as outfile:
                outfile.write(spacy_to_naf.NAF_to_string(NAF=root))

def add_ext_references_to_naf(all_docs, source_id, entity_layer_str, in_naf_dir, out_naf_dir=None):
    """Add external references (identity info) to NAF based on python objects information."""
    if out_naf_dir is not None:
        if out_naf_dir.exists():
            shutil.rmtree(str(out_naf_dir))
        out_naf_dir.mkdir()

    for news_item in all_docs:

        docid = news_item.identifier
        infile = in_naf_dir / f'{docid}.naf'
        parser = etree.XMLParser(remove_blank_text=True)
        naf_file = etree.parse(infile, parser)

        root = naf_file.getroot()
        entities_layer = root.find(entity_layer_str)

        eid2identity = {}
        entities=news_item.sys_entity_mentions
        for e in entities:
            eid2identity[e.eid] = e.identity
        print(docid, eid2identity, len(entities))
        for naf_entity in entities_layer.findall('entity'):
            eid = naf_entity.get('id')
            identity = eid2identity[eid]
            ext_refs = naf_entity.find('externalReferences')
            ext_ref = etree.SubElement(ext_refs, 'externalRef')
            ext_ref.set('reference', identity)
            ext_ref.set('source', source_id)

        if out_naf_dir is not None:
            outfile_path = out_naf_dir / f'{docid}.naf'
            with open(outfile_path, 'w') as outfile:
                outfile.write(spacy_to_naf.NAF_to_string(NAF=root))


