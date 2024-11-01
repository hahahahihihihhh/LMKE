# 导入模型
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, BatchEncoding
from transformers.models.megatron_gpt2.checkpoint_reshaping_and_interoperability import transformers_to_megatron

from model import LMKE

time_slot = '2015_1_1_0_00'


params_path = './params/SZ-Taxi-{}-bert_tiny-desc-batch_size=64-prefix_tuning=False-max_desc_length=256-epc_13_metric_fil_hits10.pt'.format(time_slot)
model_path = "./cached_model/models--{}".format('prajjwal1/bert-tiny')

lm_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
lm_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
lm_model = AutoModel.from_pretrained(model_path, config=lm_config, local_files_only=True)

in_paths = {
			'entity2text': './data/SZ-Taxi/{}/entity2text.txt'.format(time_slot),
			'relation2text': './data/SZ-Taxi/{}/relation2text.txt'.format(time_slot),
}


ent2id, er2tokens = {}, {}

with open(in_paths['entity2text'], 'r', encoding='utf8') as fil:
    id = 0
    for line in fil.readlines():
        ent, text = line.strip('\n').split('\t', 1)
        ent2id[ent] = id
        tokens = lm_tokenizer.tokenize(text)
        er2tokens[ent] = tokens
        id += 1

rel2id = {}
with open(in_paths['relation2text'], 'r', encoding='utf8') as fil:
    id = 0
    for line in fil.readlines():
        rel, text = line.strip('\n').split('\t', 1)
        rel2id[rel] = id
        tokens = lm_tokenizer.tokenize(text)
        er2tokens[rel] = tokens
        id += 1


n_ent, n_rel = len(ent2id), len(rel2id)
ent, rel = ["[ent_{}]".format(_) for _ in range(n_ent)], ["[rel_{}]".format(_) for _ in range(n_rel)]
prompt = ["[head_b1]", "[head_b2]", "[head_a1]", "[head_a2]",
			"[rel_b1]", "[rel_b2]", "[rel_a1]", "[rel_a2]",
			"[tail_b1]", "[tail_b2]", "[tail_a1]", "[tail_a2]"]

lm_tokenizer.add_tokens(prompt + ent + rel)
lm_model.resize_token_embeddings(len(lm_tokenizer))
model = LMKE(lm_model, n_ent = len(ent2id), n_rel = len(rel2id), add_tokens = True, contrastive = True)

checkpoint = torch.load(params_path)
    # 仅加载模型的 state_dict
model.load_state_dict(checkpoint['model_state_dict'])



def my_tokenize(tokens):
    '''
        start_tokens = ['[CLS]']
        end_tokens = ['[SEP]']
        pad_token = '[PAD]'
    '''
    start_tokens = [lm_tokenizer.cls_token]
    end_tokens = [lm_tokenizer.sep_token]

    tokens = [start_tokens + _ + end_tokens for _ in tokens]
    tokens_size = len(tokens)
    longest = max([len(_) for _ in tokens])

    input_ids = torch.zeros((tokens_size, longest)).long()
    token_type_ids = torch.zeros((tokens_size, longest)).long()
    attention_mask = torch.zeros((tokens_size, longest)).long()

    for _ in range(tokens_size):
        _tokens = lm_tokenizer.convert_tokens_to_ids(tokens[_])
        input_ids[_, :len(_tokens)] = torch.tensor(_tokens).long()
        attention_mask[_, :len(_tokens)] = 1

    return BatchEncoding(data={'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})


def element_to_text(target):
    text_tokens = er2tokens[target]

    if target in ent2id.keys():
        token = ['[ent_{}]'.format(ent2id[target])]
    else:
        token = ['[rel_{}]'.format(rel2id[target])]

    tokens = token + text_tokens
    text = lm_tokenizer.convert_tokens_to_string(tokens)

    return text, tokens

def tokenize(targets):
    texts = []
    tokens = []
    positions = []
    for _target in targets:
        text, token = element_to_text(_target)
        texts.append(text)
        tokens.append(token)

    tokens = my_tokenize(tokens)

    for _i, _ in enumerate(tokens['input_ids']):
        target = targets[_i]
        target_pos = 1
        if target in ent2id.keys():
            target_idx = ent2id[target]
        else:
            target_idx = rel2id[target]

        positions.append((target_idx, target_pos))
    return tokens, positions


ents, rels = list(ent2id.keys()), list(rel2id.keys())
ent_inputs, ent_positions = tokenize(targets=ents)
rel_inputs, rel_positions = tokenize(targets=rels)
print(rel_inputs, rel_positions)
ent_encodes = model.encode_target(ent_inputs, ent_positions, mode = None)
rel_encodes = model.encode_target(rel_inputs, rel_positions, mode = None)
print(ent_encodes, ent_encodes.shape)
print(rel_encodes, rel_encodes.shape)



# ['[CLS]', '[ent_0]', 'a', 'place', 'for', 'temporary', 'lodging', '.', '[SEP]'] [101, 30690, 1037, 2173, 2005, 5741, 26859, 1012, 102]
# ['[CLS]', '[rel_0]', 'adjacent', '[SEP]'] [101, 30856, 5516, 102]



