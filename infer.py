import torch
import sys
sys.path.append("/home/mdisk2/tanjunwen/gitprj/transformer")
from own_realize.models import Transformer
import torch.nn as nn

from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset

def read_sentences(file_path):
    with open(file_path, encoding='utf-8') as file:
        sentences = file.read().splitlines()
    return sentences
train_de = read_sentences('/home/mdisk2/tanjunwen/gitprj/transformer/own_realize/data/train.de')
train_en = read_sentences('/home/mdisk2/tanjunwen/gitprj/transformer/own_realize/data/train.en')
val_de = read_sentences('/home/mdisk2/tanjunwen/gitprj/transformer/own_realize/data/val.de')
val_en = read_sentences('/home/mdisk2/tanjunwen/gitprj/transformer/own_realize/data/val.en')
test_de = read_sentences('/home/mdisk2/tanjunwen/gitprj/transformer/own_realize/data/test.de')
test_en = read_sentences('/home/mdisk2/tanjunwen/gitprj/transformer/own_realize/data/test.en')


tokenizer_de = get_tokenizer('spacy', language='de_core_news_sm')
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens(data_iter, tokenizer):
    for sentence in data_iter:
        yield tokenizer(sentence)

vocab_de = build_vocab_from_iterator(yield_tokens(train_de, tokenizer_de), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
vocab_en = build_vocab_from_iterator(yield_tokens(train_en, tokenizer_en), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

vocab_de.set_default_index(vocab_de['<unk>'])
vocab_en.set_default_index(vocab_en['<unk>'])

from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)  # 解压批次数据为源语言和目标语言列表

    # 对批次中的序列进行填充
    src_batch_padded = pad_sequence(src_batch, padding_value=src_pad_idx, batch_first=True)
    trg_batch_padded = pad_sequence(trg_batch, padding_value=trg_pad_idx, batch_first=True)
    
    return src_batch_padded, trg_batch_padded


from torch.utils.data import DataLoader, Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab, tokenizer_src, tokenizer_trg):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        trg_sentence = self.trg_sentences[idx]
        src_tokens = ['<bos>'] + self.tokenizer_src(src_sentence) + ['<eos>']
        trg_tokens = ['<bos>'] + self.tokenizer_trg(trg_sentence) + ['<eos>']
        src_tensor = torch.tensor([self.src_vocab[token] if token in self.src_vocab else self.src_vocab['<unk>'] for token in src_tokens], dtype=torch.long)
        trg_tensor = torch.tensor([self.trg_vocab[token] if token in self.trg_vocab else self.trg_vocab['<unk>'] for token in trg_tokens], dtype=torch.long)
        return src_tensor, trg_tensor

train_dataset = TranslationDataset(train_de, train_en, vocab_de, vocab_en, tokenizer_de, tokenizer_en)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


test_dataset = TranslationDataset(test_de, test_en, vocab_de, vocab_en, tokenizer_de, tokenizer_en)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# 示例参数定义
src_vocab_size = len(vocab_de)  # 假设源词汇表大小
trg_vocab_size = len(vocab_en)  # 假设目标词汇表大小
src_pad_idx = vocab_de['<pad>']  # 假设源序列填充索引
trg_pad_idx = vocab_en['<pad>']  # 假设目标序列填充索引
embed_size = 512  # 嵌入层大小
num_layers = 6  # Transformer层数
heads = 8  # 多头注意力头数
device = "cuda:4" if torch.cuda.is_available() else "cpu"
max_length = 100  # 序列最大长度
model = Transformer(src_vocab_size, 
                    trg_vocab_size, 
                    src_pad_idx, 
                    trg_pad_idx, 
                    embed_size=embed_size,
                    num_layers=num_layers,
                    heads=heads,
                    device=device,
                    max_length=max_length)
# 加载模型权重
model.load_state_dict(torch.load('/home/mdisk2/tanjunwen/gitprj/transformer/own_realize/saved_models/50.pt'))
model.to(device)
model.eval()

from tqdm import tqdm
predict_sen_ls = []
for batch_idx, (src, trg) in tqdm(enumerate(test_dataloader)):
    src, trg = src.to(device), trg.to(device)
    output = model(src, trg[:, :-1])  # 输入到模型
    for i in range(src.shape[0]):
        output_ls = []
        trg_ls = []
        output_tokens = output.argmax(2)[i]
        for output_token in output_tokens:
            output_ls.append(vocab_en.get_itos()[output_token.item()])
        # trg_tokens = trg[i]
        # for trg_token in trg_tokens:
        #     if trg_token!=vocab_en['<pad>']:
        #         trg_ls.append(vocab_en.get_itos()[trg_token.item()])
        
        predict_sen = " ".join(output_ls)
        # label_sen = " ".join(trg_ls[1:-1])
        # print("\n","Predicted:", predict_sen)
        # print("\n","Label:", label_sen)
        predict_sen_ls.append(predict_sen)
    # break
    

import re
def remove_repeated_dots(sentence):
    # 正则表达式匹配一个点后面跟着一个或多个点的模式
    return re.sub(r'\.(\s*\.)+', '.', sentence)

# 假设 predict_sen_ls 是一个包含所有预测句子的列表
predict_sen_ls = [remove_repeated_dots(sent) for sent in predict_sen_ls]

trg_sen_ls = test_en
from nltk.translate.bleu_score import corpus_bleu
# 假设您已经有了所有预测句子(predict_sen_ls)和对应的标签句子
references = [[ref.lower().split()] for ref in trg_sen_ls]  # 真实目标句子
candidates = [pred.split() for pred in predict_sen_ls]  # 预测句子
# 计算BLEU得分
bleu_score = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
print(f'BLEU score: {bleu_score*100:.2f}')

# 确保保存模型的路径存在
result_folder = "/home/mdisk2/tanjunwen/gitprj/transformer/own_realize/result/"
result_path = result_folder + "test_result.txt"
import os
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
with open(result_path,"w") as f:
    for predict_sen in predict_sen_ls:
        f.write(predict_sen + "\n")