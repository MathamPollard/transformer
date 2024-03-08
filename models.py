import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    #输入维度应该是[batch_size, num_heads, seq_len, depth]
    #而不是[batch_size, seq_len, num_heads, depth]
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))#转置最后两个维度，计算qk相似度
    d_k = q.size(-1) #获取查询向量q的最后一个维度的大小，即键向量的维度d_k。这个值用于后面的缩放操作。
    scaled_attention_logits = matmul_qk / math.sqrt(d_k)
    #将点积结果除以d_k的平方根进行缩放。这样做是为了控制点积结果的量级，防止在d_k值很大时梯度消失或爆炸
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)#计算每个查询对所有键的注意力权重。dim=-1指定在最后一个维度上执行softmax
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = torch.nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = torch.nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = torch.nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = torch.nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, queries, mask):
        # queries = queries.transpose(0, 1)
        # keys = keys.transpose(0, 1)
        # values = values.transpose(0, 1)
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        # N = queries.shape[1]
        # value_len, key_len, query_len = values.shape[0], keys.shape[0], queries.shape[0]

        # 假设输入已经是 [N, seq_len, embed_size]
        queries = self.queries(queries).reshape(N, query_len, self.heads, self.head_dim)
        keys = self.keys(keys).reshape(N, key_len, self.heads, self.head_dim)
        values = self.values(values).reshape(N, value_len, self.heads, self.head_dim)


        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # Calculate the attention using the scaled dot product
        attention, _ = scaled_dot_product_attention(queries, keys, values, mask)

        # Concatenate heads and put it through final linear layer
        attention = attention.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(attention)
        
        return out

class EncoderLayer(torch.nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = torch.nn.LayerNorm(embed_size)
        self.norm2 = torch.nn.LayerNorm(embed_size)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_size, forward_expansion * embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = torch.nn.Dropout(dropout)

        self.device = device

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Apply dropout and add & norm
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #生成一个从0到max_len的连续整数向量，表示位置索引，然后通过unsqueeze(1)将其变成二维矩阵，方便后续的计算。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #计算位置编码的分母项。这里使用指数函数和对数函数来生成按位置索引递减的系数，这些系数用于调整正弦和余弦函数的频率。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #将pe矩阵的形状从[max_len, d_model]变更为[1, max_len, d_model]，并转置为[max_len, 1, d_model]，
        # 使其符合(batch_size, sequence_length, d_model)的形状要求。
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe[:x.size(0), :]
        # return x
        # return self.pe[:x.size(0), :]
        return self.pe[:x.size(1), :]

class Encoder(torch.nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        #为了能够以批处理的方式同时处理多个序列，通常需要将所有序列填充（pad）到相同的长度，即max_length
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = torch.nn.Embedding(src_vocab_size, embed_size)
        #基本是把src_vocab_size维向量映射为embed_size维向量的线性层而已
        # self.position_embedding = torch.nn.Embedding(max_length, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)
        #前馈神经网络的中间层维度是输入层维度（embed_size）的forward_expansion倍。
        # 例如，如果embed_size为512，forward_expansion为4，则FFN的中间层维度将是2048
        self.layers = torch.nn.ModuleList([
            EncoderLayer(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
        ])

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        # N, seq_length = x.shape
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        out = self.dropout(self.word_embedding(x) + self.position_embedding(x).transpose(0, 1))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        #value, key, query都是out，自身与自身的注意力，这就是“自注意力”的意思
        return out

class DecoderLayer(torch.nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        #self.attention子层是解码器的掩蔽多头自注意力机制
        #self.transformer_block用于处理来自编码器的输出
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm = torch.nn.LayerNorm(embed_size)
        self.transformer_block = MultiHeadAttention(embed_size, heads)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_size, forward_expansion * embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(embed_size)
        self.norm2 = torch.nn.LayerNorm(embed_size)
        self.norm3 = torch.nn.LayerNorm(embed_size)
        self.device = device

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        # out = self.transformer_block(query, key, value, src_mask)
        out = self.transformer_block( value, key, query, src_mask)
        transformer_out = self.dropout(self.norm1(out + query))
        feed_forward_out = self.feed_forward(transformer_out)
        out = self.dropout(self.norm2(feed_forward_out + transformer_out))
        return out

class Decoder(torch.nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = torch.nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)

        self.layers = torch.nn.ModuleList([
            DecoderLayer(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
        ])

        # self.fc_out = torch.nn.Linear(embed_size, trg_vocab_size)
        #把词向量重新映射为代表单词的整数空间
        #计算损失就是用这个的输出和真实标签算的
        #这一行代码定义的是一个线性层（torch.nn.Linear），它在Transformer模型的解码器部分的末尾
        #被用来将解码器的输出映射到一个更大的空间——目标语言的词汇空间。这个层的目的是为了预测每个位置上的下一个可能的单词。
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # N, seq_length = x.shape
        # x = self.dropout(self.word_embedding(x) + self.position_embedding(x))
        x = self.dropout(self.word_embedding(x) + self.position_embedding(x).transpose(0, 1))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        # out = self.fc_out(x)
        # return out
        return x

class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0.1, device="cuda", max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.fc_out = torch.nn.Linear(embed_size, trg_vocab_size)
        self.dropout = torch.nn.Dropout(dropout)
    #trg_mask是一个下三角形矩阵。这种特殊的掩码结构用于确保在生成当前单词时，
    # 模型只能依赖于之前的单词（包括当前位置），而不能"看到"未来的单词。
    # 这是实现序列生成任务中自回归性质的关键
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = (src != self.src_pad_idx).unsqueeze(-1).unsqueeze(-1)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        out = self.fc_out(out)
        return out

