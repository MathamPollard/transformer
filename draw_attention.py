import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))#转置最后两个维度，计算qk相似度
    d_k = q.size(-1) #获取查询向量q的最后一个维度的大小，即键向量的维度d_k。这个值用于后面的缩放操作。
    scaled_attention_logits = matmul_qk / math.sqrt(d_k)
    #将点积结果除以d_k的平方根进行缩放。这样做是为了控制点积结果的量级，防止在d_k值很大时梯度消失或爆炸
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)#计算每个查询对所有键的注意力权重。dim=-1指定在最后一个维度上执行softmax
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

import numpy as np
import matplotlib.pyplot as plt
if __name__=="__main__":
    # 设置随机种子以便结果可复现
    torch.manual_seed(0)
    np.random.seed(0)

    # 参数定义
    # seq_len = 10  # 序列长度
    # d_k = 64  # 向量维度
    # batch_size = 1  # 批大小
    # 生成Q, K, V
    # q = torch.rand((batch_size, seq_len, d_k))
    # k = torch.rand((batch_size, seq_len, d_k))
    # v = torch.rand((batch_size, seq_len, d_k))
    seq_len = 5  # 序列长度
    d_k = 5  # 向量维度
    batch_size = 1  # 批大小

    # 生成Q, K, V
    q_ls = [[
        [0.1,0.2,0.3,0.4,0.5],
        [0.1,0.2,0.3,0.4,0.5],
        [0.1,0.2,0.3,0.4,0.5],
        [0.5,0.5,0.5,0.5,0.5],
        [0.1,0.2,0.3,0.4,0.5],
    ]]
    k_ls = [[
        [0.1,0.2,0.3,0.4,0.5],
        [0.2,0.2,0.2,0.2,0.2],
        [0.3,0.3,0.3,0.3,0.3],
        [0.4,0.4,0.4,0.4,0.4],
        [0.5,0.4,0.3,0.2,0.1]
    ]]
    q = torch.tensor(q_ls)
    k = torch.tensor(q_ls)
    v = torch.rand((batch_size, seq_len, d_k))
    
    #测试函数并可视化注意力权重
    output, attention_weights = scaled_dot_product_attention(q, k, v)

    # 可视化注意力权重
    plt.imshow(attention_weights[0].detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.title('Attention Weights')
    plt.savefig('attention_weights.png')  # 保存图像
    plt.show()
