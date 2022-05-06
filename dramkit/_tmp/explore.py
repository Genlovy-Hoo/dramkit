# -*- coding: utf-8 -*-

# https://blog.csdn.net/david0611/article/details/81090294

#%%
if __name__ == '__main__':

    #%%
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable as V

    #%%
    # 线性层
    line=nn.Linear(2, 4) # 输入2维，输出4维
    print(line.weight) # 参数是随机初始化的，维度为out_dim*in_dim
    print(line.weight.shape)
    x=V(torch.randn(5, 2)) # batch为5
    print(x)
    print(x.shape)
    x_out=line(x)
    print(x_out) # 输出为batch*4
    print(x_out.shape)

    #%%
    # RNN层
    input_size=5
    hidden_size=8
    num_layers=4
    # 构造RNN网络，x的维度input_size，隐层的维度hidden_size，网络的层数num_layers
    rnn_seq=nn.RNN(input_size, hidden_size, num_layers)
    # 构造一个输入序列，长为6，batch是3，特征是5
    x=V(torch.randn(6,3,input_size)) # 3个样本，样本序列长度为6，每个样本维度为input_size
                                     # 对文本相当于：3句话，每句话长度为6个词，每个词维度为input_size
                                     # 对时间序列相当于：3个样本，每个样本长度为6，总共有input_size个特征（变量）
    out,ht=rnn_seq(x) # h0可以指定或者不指定
    # q1: 这里out、ht的size是多少呢？out: 6*3*hidden_size，ht: num_layers*3*hidden_size
    print(out.size()) #（序列长度*样本数*隐藏层维度）
    print(ht.size())  #（隐藏层数*样本数*隐藏层维度）
    # q2: out[-1]和ht[-1]是否相等？相等！
    print(out[-1] == ht[-1]) # out保存的是最后个一层隐藏层的序列状态，除了特征维度与输入不一样（变成了隐藏层维度），其它形状是一样的
                             # out[-1]即序列中的最后一个状态（相当于句子的最后一个词或时间序列的最后一个时刻）
                             # 如果仅使用out[-1]，则相当于假设把前面时刻的信息都编码进最后一个时刻了（用序列的最后一个状态来表示前面所有状态的编码信息）
                             # ht保存的是每个隐藏层的最后一个状态（即保存了每个隐藏层中序列的最后时刻的值，所以out[-1]与ht[-1]相等，因为两者都是最后一个隐藏层中序列的最后一个状态值）

    #%%
    # RNN层（双向）
    input_size=5
    hidden_size=8
    num_layers=4
    # 构造RNN网络，x的维度input_size，隐层的维度hidden_size，网络的层数num_layers
    rnn_seq=nn.RNN(input_size,hidden_size,num_layers,bidirectional=True)
    # 构造一个输入序列，长为6，batch是3，特征是5
    x=V(torch.randn(6,3,input_size)) # 3个样本，样本序列长度为6，每个样本维度为input_size
                                     # 对文本相当于：3句话，每句话长度为6个词，每个词维度为input_size
                                     # 对时间序列相当于：3个样本，每个样本长度为6，总共有input_size个特征（变量）
    out,ht=rnn_seq(x) # h0可以指定或者不指定
    # q1: 这里out、ht的size是多少呢？out: 6*3*(hidden_size*2)，ht: (num_layers*2)*3*hidden_size
    print(out.size()) #（序列长度*样本数*隐藏层维度*2），因为是双向网络，所以输出是两个方向结果的拼接，故输出维度要是输入维度乘2
    print(ht.size())  #（2*隐藏层数*样本数*隐藏层维度），因为是双向网络，所以记忆单元保存了两个方向的结果，故隐藏层数*2
                      # 也就是说序列输出时对双向结果进行了拼接，但是记忆单元保存的结果是没有拼接的
    # q2: out[-1]和ht[-1]是否相等？不相等！但是out[-1]的前半部分等于ht[-2]，out[0]后半部分等于ht[-1]
    print(torch.cat((out[-1,:,0:hidden_size],out[0,:,hidden_size:]),dim=1) == torch.cat((ht[-2],ht[-1]),dim=1))
    print(out[-1] == torch.cat((ht[-2],ht[-1]),dim=1))
    print(out[0] == torch.cat((ht[-2],ht[-1]),dim=1))
    # out保存的是最后个一层隐藏层的序列状态，除了特征维度与输入不一样（由于双向拼接，所以特征维度变为hidden_size*2），其它形状是一样的
    # out[-1]即序列中的正向最后一个状态（对应时间0—>t的时刻t）和反向第一个状态（对应时间t—>0的时刻t）的拼接
    # ht保存的是每个隐藏层每个方向的最后一个状态（正向时对应时间0—>t的时刻t，反向时对应时间t->0的时刻0）

    #%%
    # RNN层（batch_first）
    input_size=5
    hidden_size=8
    num_layers=2
    # 构造RNN网络，x的维度input_size，隐层的维度hidden_size，网络的层数num_layers
    rnn_seq=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    # 构造一个输入序列，长为6，batch是3，特征是5
    x=V(torch.randn(3,6,input_size)) # 3个样本，样本序列长度为6，每个样本维度为input_size
                                     # 对文本相当于：3句话，每句话长度为6个词，每个词维度为input_size
                                     # 对时间序列相当于：3个样本，每个样本长度为6，总共有input_size个特征（变量）
    out,ht=rnn_seq(x) # h0可以指定或者不指定
    # q1: 这里out、ht的size是多少呢？out: 3*6*hidden_size，ht: num_layers*3*hidden_size
    print(out.size()) #（序列长度*样本数*隐藏层维度）
    print(ht.size())  #（隐藏层数*样本数*隐藏层维度）
    # q2: out[-1]和ht[-1]是否相等？相等！
    print(out[:,-1,:] == ht[-1]) # out保存的是最后个一层隐藏层的序列状态，除了特征维度与输入不一样（变成了隐藏层维度），其它形状是一样的
                                 # out[:,-1,:]即序列中的最后一个状态（相当于句子的最后一个词或时间序列的最后一个时刻）
                                 # 如果仅使用out[:,-1,:]，则相当于假设把前面时刻的信息都编码进最后一个时刻了（用序列的最后一个状态来表示前面所有状态的编码信息）
                                 # ht保存的是每个隐藏层的最后一个状态（即保存了每个隐藏层中序列的最后时刻的值，所以out[-1]与ht[-1]相等，因为两者都是最后一个隐藏层中序列的最后一个状态值）

    #%%
    # LSTM层（batch_first）
    input_size=7
    hidden_size=9
    num_layers=4
    # 输入维度input_size，隐层维度hidden_size，层数num_layers
    lstm_seq=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
    # 查看网络的权重，ih和hh，共2层，所以有四个要学习的参数
    print((lstm_seq.weight_hh_l0.size(),
           lstm_seq.weight_hh_l1.size(),
           lstm_seq.weight_ih_l0.size(),
           lstm_seq.weight_ih_l1.size()))
    # 输入序列seq=10，batch=3，输入维度=input_size
    lstm_input=V(torch.randn(3,10,input_size))
    out,(h,c)=lstm_seq(lstm_input) # 使用默认的全 0 隐藏状态
    # q1：out和(h,c)的size各是多少？out：(3*10*hidden_size)，（h,c）：都是(num_layers*3*100)
    print(out.size()) # out保存的是最后个一层隐藏层的序列状态，除了特征维度与输入不一样（变成了隐藏层维度），其它形状是一样的
    print(h.size()) # h保存的是每个隐藏层的最后一个状态（即保存了每个隐藏层中序列的最后时刻的值，所以out[:,-1,:]与h[-1]相等，因为两者都是最后一个隐藏层中序列的最后一个状态值）
    print(c.size()) # c保存的是每个隐藏层最后一个状态的记忆单元，其形状与h一致
    # q2：out[:,-1,:]和h[-1,:,:]相等吗？相等
    print(out[:,-1,:] == h[-1])

    #%%
    # GRU层（batch_first）
    input_size=7
    hidden_size=9
    num_layers=4
    # 输入维度input_size，隐层维度hidden_size，层数num_layers
    gru_seq=nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
    gru_input=V(torch.randn(3,10,input_size))# 输入序列seq=10，batch=3，输入维度=input_size
    out,h=gru_seq(gru_input)
    # GRU的输出形状跟标准RNN完全一样
    print(out.size())
    print(h.size())
    print(out[:,-1,:] == h[-1])

    #%%
    class SelfAttention(nn.Module):
        def __init__(self,hidden_dim):
            super().__init__()
            self.hidden_dim=hidden_dim
            self.projection=nn.Sequential(nn.Linear(hidden_dim,64),
                                          nn.ReLU(True),
                                          nn.Linear(64,1))
        def forward(self,encoder_outputs):
            # (batch_size,len_seq,features_num)->(batch_size,1)
            energy=self.projection(encoder_outputs)
            # 权重（序列中每个状态（时刻）的权重）
            weights=F.softmax(energy.squeeze(-1),dim=1) #
            # (B,L,H)*(B,L,1)->(B,H)
            outputs=(encoder_outputs*weights.unsqueeze(-1)).sum(dim=1)
            return outputs,weights

    Self_Attn=SelfAttention(hidden_dim=hidden_size)
    out_selfattn,weights=Self_Attn(out)
    print(out_selfattn.size())
    print(weights.size())

    #%%
    class AttnClassifier(nn.Module):
        def __init__(self,input_dim,embedding_dim,hidden_dim):
            super().__init__()
            self.input_dim=input_dim
            self.embedding_dim=embedding_dim
            self.hidden_dim=hidden_dim
            self.embedding=nn.Embedding(input_dim,embedding_dim)
            self.lstm=nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
            self.attention=SelfAttention(hidden_dim)
            self.fc=nn.Linear(hidden_dim,1)

        def set_embedding(self,vectors):
            self.embedding.weight.data.copy_(vectors)

        def forward(self,inputs,lengths):
            batch_size=inputs.size(1)
            # (L,B)
            embedded=self.embedding(inputs)
            # (L,B,E)
            packed_emb=nn.utils.rnn.pack_padded_sequence(embedded,lengths)
            out,hidden=self.lstm(packed_emb)
            out=nn.utils.rnn.pad_packed_sequence(out)[0]
            out=out[:,:,:self.hidden_dim]+out[:,:,self.hidden_dim:]
            # (L,B,H)
            embedding,attn_weights=self.attention(out.transpose(0,1))
            # (B,HOP,H)
            outputs=self.fc(embedding.view(batch_size,-1))
            # (B,1)
            return outputs,attn_weights
