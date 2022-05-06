# -*- coding: utf-8 -*-

# https://wmathor.com/index.php/archives/1455/

if __name__ == '__main__':

    import math
    import torch
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as Data

    #%%
    # 数据准备

    # S: Symbol that shows starting of decoding input
    # S: 解码器输入开始标识
    # E: Symbol that shows starting of decoding output
    # E: 解码器输出开始标识
    # P: Symbol that will fill in blank sequence if current batch data size is
    # short than time steps
    # P: 待填充为空白字符标识？
    sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
    ]

    # 单个词（字）分配序号 Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6,
                 'E': 7, '.': 8}
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5 # enc_input max sequence length
    tgt_len = 6 # dec_input(=dec_output) max sequence length

    def make_data(sentences):
        '''原始词（字）转化为编号'''
        enc_inputs, dec_inputs, dec_outputs = [], [], []
        for i in range(len(sentences)):
            # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
            enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
            # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
            dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
            # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
            dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

            enc_inputs.extend(enc_input)
            dec_inputs.extend(dec_input)
            dec_outputs.extend(dec_output)

        enc_inputs = torch.LongTensor(enc_inputs)
        dec_inputs = torch.LongTensor(dec_inputs)
        dec_outputs = torch.LongTensor(dec_outputs)

        return enc_inputs, dec_inputs, dec_outputs

    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

    class MyDataSet(Data.Dataset):
        '''构造torch数据集'''
        def __init__(self, enc_inputs, dec_inputs, dec_outputs):
            super(MyDataSet, self).__init__()
            self.enc_inputs = enc_inputs
            self.dec_inputs = dec_inputs
            self.dec_outputs = dec_outputs

        def __len__(self):
            return self.enc_inputs.shape[0]

        def __getitem__(self, i):
            return self.enc_inputs[i], self.dec_inputs[i], self.dec_outputs[i]

    mydata = MyDataSet(enc_inputs, dec_inputs, dec_outputs)
    loader = Data.DataLoader(mydata, batch_size=2, shuffle=True)

    #%%
    # 参数

    d_model = 512 # 词向量&位置向量维度 Embedding Size
    d_ff = 2048 # FeedForward全连接层隐藏神经元个数 FeedForward dimension
    d_k = d_v = 64 # Q、K、V 向量的维度 dimension of K(=Q), V
    n_layers = 6 # Encoder 和 Decoder 的个数 number of Encoder and Decoder Layer
    # 多头注意力中 head 的数量 number of heads in Multi-Head Attention
    # 注：n_heads = d_model / d_v？
    n_heads = 8

    #%%
    # 位置编码

    class PositionalEncoding(nn.Module):
        '''位置编码'''
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            '''
            max_len: ？
            '''
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            # unsqueeze维度扩展
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            # 1 / (10000 ^ (2i / dmodel)) = 10000 ^ (-2i / dmodel) =
            # e ^ ((-2i / dmodel) * ln(10000)) （公式：a ^ b = e ^ (b * ln(a))）
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
                                 (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1) # [seq_len, batch_size, d_model]？
            self.register_buffer('pe', pe)

        def forward(self, x):
            '''
            x: [seq_len, batch_size, d_model]
            '''
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)


    def get_attn_pad_mask(seq_q, seq_k):
        '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
        '''
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        # [batch_size, 1, len_k], False is masked
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
        # [batch_size, len_q, len_k]
        return pad_attn_mask.expand(batch_size, len_q, len_k)

    # test
    seq_q = seq_k = torch.Tensor([[1, 0, 3, 4, 0]])
    print(get_attn_pad_mask(seq_q, seq_k))
