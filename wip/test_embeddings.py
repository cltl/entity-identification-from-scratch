import embeddings as emb
from pytorch_pretrained_bert import BertModel, BertTokenizer
from config import Config


def test_new_mapping(sentence, verbose=True):
    cfg = Config('../tests/cfg/test.yml')
    model = BertModel.from_pretrained(cfg.bert_model)
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model,
                                              do_lower_case=False)
    word_embeddings, sentence_embeddings, bert_tokens = emb.get_bert_word_and_sentence_embeddings(model, sentence, tokenizer, verbose)
    console.log(word_embeddings.shape)


sentence="After 5 months and 48 games , the match was abandoned in controversial circumstances with Karpov leading five wins to three ( with 40 draws ) , and replayed in the World Chess Championship 1985 ."
test_new_mapping(sentence)

