class EntityMention:
    """
    Class containing information about an entity mention
    """

    def __init__(self,
                 mention,
                 begin_index='',
                 end_index='',
                 begin_offset='',
                 end_offset='',
                 eid='-1',
                 the_type=None,
                 sentence=None,
                 identity=None,
                 tokens=[]):
        self.eid = eid
        self.sentence = sentence  # e.g. 4 -> which sentence is the entity mentioned in
        self.mention = mention  # e.g. "John Smith" -> the mention of an entity as found in text
        self.the_type = the_type  # e.g. "Person" | "http://dbpedia.org/ontology/Person"
        self.begin_index = begin_index  # e.g. 3 -> begin token id
        self.end_index = end_index  # e.g. 5 -> end token id
        self.begin_offset = begin_offset  # e.g. 15 -> begin offset
        self.end_offset = end_offset  # e.g. 25 -> end offset
        self.identity = identity  # gold link if existing
        self.tokens = tokens  # list of token ids


class NewsItem:
    """
    Class containing information about a news item
    """

    def __init__(self,
                 identifier,
                 content="",
                 collection=None,
                 title='',
                 dct=None,
                 sys_entity_mentions=[],
                 gold_entity_mentions=[]):
        self.identifier = identifier  # string, the original document name in the dataset
        self.collection = collection  # which collection does it come from
        self.dct = dct  # e.g. "2005-05-14T02:00:00.000+02:00" -> document creation time
        self.content = content  # the text of the news article
        self.title = title
        self.sys_entity_mentions = sys_entity_mentions  # set of instances of EntityMention class
        self.gold_entity_mentions = gold_entity_mentions  # set of instances of EntityMention class
