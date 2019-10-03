from lxml import etree
import glob
import datetime
import shutil
import pathlib

import spacy_to_naf

import config


def patch_classes_with_tokens(news_items, naf_dir, entity_layer):
    for item in news_items:
        docid = item.identifier
        naf_output_path = naf_dir / f'{docid}.naf'
        eid_to_tids = obtain_entity_data(naf_output_path, entity_layer)
        for e in item.sys_entity_mentions:
            eid = e.eid
            e.tokens = eid_to_tids[eid]
    return news_items


def load_sentences(naf_dir, iteration, modify_entities=False):
    all_sent = []
    for f in glob.glob('%s/*.naf' % naf_dir):
        parser = etree.XMLParser(remove_blank_text=True)
        doc = etree.parse(f, parser)

        root = doc.getroot()
        s = load_sentences_from_naf(iteration, root, modify_entities)
        all_sent += s
    return all_sent


# ------ NAF processing utils --------------------

def create_nafs(naf_folder,
                news_items,
                nl_nlp,
                corpus_uri,
                ner_system='auto',
                layers={'raw', 'text', 'terms'}):
    """Create NAFs if not there already."""

    if naf_folder.exists():
        file_count = len(glob.glob('%s/*.naf' % naf_folder))
        assert file_count == len(
            news_items), 'NAF directory exists, but a wrong amount of files. Did you edit the source documents?'
        print('NAF files were already there, and at the right number.')
        # shutil.rmtree(str(naf_folder))
    else:
        print('No NAF files found. Let\'s create them.')
        pathlib.Path(naf_folder).mkdir(parents=True, exist_ok=True)
        if ner_system != 'gold':
            layers.add('entities')
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


def add_ext_references_to_naf(all_docs, source_id, entity_layer_str, in_naf_dir, out_naf_dir=None):
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
            eid2identity = {}
            for e in entities:
                eid2identity[e.eid] = e.identity
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


def load_sentences_from_naf(iteration, root, naf_entity_layer, modify_entities):
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
            if modify_entities:
                if target != span.findall('target')[-1]:
                    to_replace[t] = ''
                else:
                    to_replace[t] = the_id

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


def obtain_entity_data(naf_file, entities_layer_id):
    parser = etree.XMLParser(remove_blank_text=True)
    doc = etree.parse(naf_file, parser)

    root = doc.getroot()

    entities_layer = root.find(entities_layer_id)
    eid_to_tids = {}
    for entity in entities_layer.findall('entity'):
        eid = entity.get('id')
        refs = entity.find('references')
        span = refs.find('span')
        tids = []
        for target in span.findall('target'):
            tids.append(target.get('id'))
        eid_to_tids[eid] = tids
    return eid_to_tids


def copy_nafs(src, tgt):
    if tgt.exists():
        shutil.rmtree(str(tgt))
    shutil.copytree(src, tgt)
