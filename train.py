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

from collections import Counter
from torchtext.vocab import Vocab

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
        src_tensor = torch.tensor([self.src_vocab[token] for token in self.tokenizer_src(src_sentence)], dtype=torch.long)
        trg_tensor = torch.tensor([self.trg_vocab[token] for token in self.tokenizer_trg(trg_sentence)], dtype=torch.long)
        return src_tensor, trg_tensor

train_dataset = TranslationDataset(train_de, train_en, vocab_de, vocab_en, tokenizer_de, tokenizer_en)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


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
model.to(device)


import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        lr = self.d_model ** (-0.5) * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))
        return [lr for base_lr in self.base_lrs]
# 使用示例
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
# scheduler = CustomScheduler(optimizer, d_model=512, warmup_steps=10000)
# 定义损失函数，忽略<pad>标记的损失计算
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    
    

import time
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
# 在训练开始时获取当前时间戳
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
log_dir = os.path.join("/home/mdisk2/tanjunwen/gitprj/transformer/own_realize/log", current_time)
writer = SummaryWriter(log_dir=log_dir)


def train_and_evaluate(model, iterator, optimizer, criterion, clip, num_epochs, model_path, device):
    model.train()  # 将模型设置为训练模式

    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time()

        for batch_idx, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()

            output = model(src, trg[:, :-1])  # 输入到模型

            output_ls = []
            trg_ls = []
            if (epoch * len(iterator) + batch_idx)%500==0:
                output_tokens = output.argmax(2)[0]
                for output_token in output_tokens:
                    output_ls.append(vocab_en.get_itos()[output_token.item()])
                trg_tokens = trg[0]
                for trg_token in trg_tokens:
                    if trg_token!=vocab_en['<pad>']:
                        trg_ls.append(vocab_en.get_itos()[trg_token.item()])
                
                print(epoch * len(iterator) + batch_idx)
                print("\n","Predicted:", " ".join(output_ls[1:-1]))
                print("\n","Label:", " ".join(trg_ls[1:-1]))
            
            # 调整输出和目标的维度，以计算损失
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            
            # 计算损失
            loss = criterion(output, trg)

            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 梯度裁剪
            optimizer.step()
            # scheduler.step()  # 更新学习率
            
            # # 获取当前学习率，如果是使用多参数组的优化器，这里获取的是第一组参数的学习率
            # current_lr = scheduler.get_last_lr()[0]
            # # 将当前学习率写入TensorBoard
            # writer.add_scalar("Learning Rate", current_lr, epoch * len(iterator) + batch_idx)
            
            epoch_loss += loss.item()

            # 将损失写入TensorBoard
            writer.add_scalar("Training Loss", loss.item(), epoch * len(iterator) + batch_idx)
        # 计算并打印平均损失
        epoch_loss /= len(iterator)
        # writer.add_scalar("Training Loss", epoch_loss, epoch)

        # 保存模型
        torch.save(model.state_dict(), f'{model_path}/{epoch + 1}.pt')

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {epoch_loss:.3f}')

    writer.close()

# 训练配置
NUM_EPOCHS = 50
CLIP = 1
MODEL_PATH = "/home/mdisk2/tanjunwen/gitprj/transformer/own_realize/saved_models"

# 确保保存模型的路径存在
import os
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 调用训练函数
train_and_evaluate(model, train_dataloader, optimizer, criterion, CLIP, NUM_EPOCHS, MODEL_PATH, device)
