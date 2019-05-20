from pprint import pprint
import spacy
from collections import Counter
import en_core_web_sm
en_nlp = en_core_web_sm.load()

import nl_core_news_sm
nl_nlp=nl_core_news_sm.load()

en_text='European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

nl_text='Google beperkt toegang Huawei tot Android-besturingssysteem. Waarschijnlijk kunnen belangrijke Google-apps en updates niet meer door Huawei worden gebruikt bij nieuwe telefoons.'

en_doc=en_nlp(en_text)
pprint([(X.start, X.end, X.text, X.label_) for X in en_doc.ents])

nl_doc=nl_nlp(nl_text)
pprint([(X.text, X.label_) for X in nl_doc.ents])
