import torch
import torch.nn as nn
import numpy as np
import logging
import torch.nn.functional as F
from utils import ShipDataset,Visualizer
from tqdm import tqdm,trange
from torch.utils import data
#########配置log日志方便打印#############

LOG_FORMAT = "%(asctime)s -%(filename)s[line:%(lineno)d]- %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m-%d-%Y %H:%M:%S"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

logger = logging.getLogger(__name__)
######################################
'''
    编码器，采用两层LSTM
'''
class encoderModel(nn.Module):
    def __init__(self,n_words,n_layers=2,input_size=8,hidden_size=10):
        super(encoderModel, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embeding = nn.Embedding(n_words,input_size,padding_idx=0)

        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.4,bidirectional=False)

    def forward(self,input,len_x,state=None):
        # 自己给自己挖坑，这里输入的input是单词的index，我现在想用可训练的词向量，也就是nn.embeding
        input = self.embeding (input)

        #这里输入的input是已经词向量化好的数据
        batch, words, vecsize = input.size()   #第一个是batchsize 第二个是单词个数 第三个是词向量的维度


        input_x = torch.nn.utils.rnn.pack_padded_sequence(input, len_x, batch_first=True)

        if state is None:
            h = torch.randn(self.n_layers, batch, self.hidden_size).float()
            c = torch.randn(self.n_layers, batch, self.hidden_size).float()
        else:
            h, c = state


        # output [batchsize,time,hidden_size]
        output, state = self.rnn(input_x, (h, c))
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        #最后输出结果
        # output = output[:, -1, :]
        output = F.log_softmax(output,dim=1)
        return output,state

'''
    一个解码器单元结构：
    解码器：使用两层lstm，加上注意力机制
    初始的隐藏状态是编码器最后LSTM的隐藏状态
    开始输入值为SOS_token = 0，结束值为EOS_token = 1。
    解码器的输入值为前一个时刻的输出和由attention权重算出的编码器上下文语义的拼接
    
    attention： score(State,output) = W*state*output  其中W为要学习的权重
    这里用一个线性层来实现
    
'''
class ArrdecoderModel(nn.Module):
    def __init__(self,n_words,n_layers=2,input_size=10,hidden_size=10):
        super(ArrdecoderModel, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size  #由于直接用编码器的state初始化解码器，所以这里的hidden_size和编码器的相等
        self.hidden_size = hidden_size
        self.embeding = nn.Embedding(n_words, input_size)
        if input_size != hidden_size:
            raise RuntimeError('解码器的前一个输出即为后一个时刻的输入，所以inputSize应该与hiddenSize一致！')

        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.2)
        self.att_line = nn.Linear(hidden_size*2,1,bias=True)
        self.input_line = nn.Linear(hidden_size + input_size, input_size,bias=True)

        self.out = nn.Linear(hidden_size,n_words,bias=True)

    def forward(self,input,state,encoder_output):

        input = self.embeding(input)



        #解码器的第一个输入是为SOS_token = 0,如果输出为EOS_token = 1
        batch, words, hidden_size = encoder_output.size()   # output [batchsize,time,hidden_size]

        if state is None:
            raise RuntimeError('解码器的state为空，请将编码器的state或者前一个时刻的state传进来！')

        '''加入attention
            将前一个状态输出的state和encoder_output进行F(X)，得到对encoder_output各输出的权重
            对encoder_output进行加权求和，得到这一时刻的输入
        '''
        #这里取最后一层的state来进行计算,将state里的h点乘c作为状态，其实也可以直接拿h或者c
        #这样得到的att_state = [batch,hidden]
        att_state = torch.mul(state[0][-1],state[1][-1])
        #初始化权重
        weigth = torch.zeros(batch, 1)
        #计算权重 这里将编码器每一个时间点的输出与att_state进行全连接得到一个权重值
        for word in range(words):
            l = torch.cat((encoder_output[:, word, :], att_state), dim=1)
            w = self.att_line(l)
            weigth = torch.cat((weigth, w), dim=1)
        weigth = F.softmax(weigth[:, 0:-1],dim=1)  #去除第一列然后进行softmax得到权重

        #将权重与encoder_output相乘
        weigth = weigth.unsqueeze(1)
        x = torch.bmm(weigth, encoder_output)  #[batch,time=1,hidden_size]
        x = torch.cat((x,input),dim=2)  #前一时刻的输出和attention计算出的上下文语义拼接，并进行全连接成[batch,time=1,hidden]
        input = self.input_line(x)


        # output [batchsize,time,hidden_size]
        output, state = self.rnn(input, state)

        output = self.out(output.squeeze()).unsqueeze(1)

        output = F.log_softmax(output,dim=2)

        return output,state

def train(input_tensor,len_x,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion):
    #原谅我这里写的和pytorch官网上一模一样，其实我实在觉得这玩意都差不多，没想到更好的封装了
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    #这里取target的长度作为解码器的输出个数限制
    target_length = target_tensor.size(1)


    encoder_outputs,state = encoder(input_tensor,len_x)

    # 这里定义解码器的开始和结束标志
    decoder_input = torch.zeros((encoder_outputs.size(0),1)).long()
    END_token = torch.ones(encoder_outputs.size(0)).long()

    for i in range(target_length):
        decoder_output,state = decoder(decoder_input,state,encoder_outputs)

        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze(1).detach()

        # decoder_input = decoder_output

        loss += criterion(torch.squeeze(decoder_output),target_tensor[:,i])#IndexError: too many indices for tensor of dimension 2
        # for j in range(encoder_outputs.size(0)):
        #     loss+= criterion(decoder_end[j],target_tensor[j,i])
        if torch.equal(decoder_input,END_token):
            break


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder,decoder,dataiters,epoch,vis,print_step = 50,lr = 0.01):
    en_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=0.0005)
    de_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=0.0005)

    criterion = nn.NLLLoss()
    sch_en = torch.optim.lr_scheduler.ReduceLROnPlateau(en_optimizer, mode='min', factor=0.1, patience=5,
                                               verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0,
                                               min_lr=0, eps=1e-08)

    sch_de = torch.optim.lr_scheduler.ReduceLROnPlateau(de_optimizer, mode='min', factor=0.1, patience=5,
                                                     verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                     min_lr=0, eps=1e-08)
    for e in trange(1,epoch+1):
        for i, trainset in enumerate(dataiters):
            x, y, len_x, n_words = trainset
            loss = train(x,len_x, y, encoder, decoder, en_optimizer, de_optimizer, criterion)

            if i %print_step ==0:
                print('batch_loss: {0}'.format(loss))
            vis.plot('loss',loss)
        print('epoch : %s ,the loss is '%{loss})
        sch_en.step(loss)
        sch_de.step(loss)

#evaluate the model
def evaluate(encoder, decoder,input_tensor,target_tensor,len_x,index2word):
    with torch.no_grad():
        encoder_outputs, state = encoder(input_tensor, len_x)
        target_length = target_tensor.size(1)
        # 这里定义解码器的开始和结束标志
        decoder_input = torch.zeros((encoder_outputs.size(0), 1)).long()
        END_token = torch.ones(encoder_outputs.size(0)).long()

        decoded_words = []
        for i in range(target_length):
            decoder_output, state = decoder(decoder_input, state, encoder_outputs)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1).detach()

            if torch.equal(decoder_input, END_token):
                break

            decoded_words.append(index2word[decoder_input.item()])

        return decoded_words

def my_collate(batch):
    '''
                    batch里面长度补齐实现策略
                    将batch按照x的长度进行排序
                    按照句子的最长长度来补齐
                '''
    batch.sort(key=lambda data:len(data[0]),reverse=True)
    data_len_x = [len(x[0]) for x in batch]
    data_len_y = [len(x[1]) for x in batch]
    n_words = batch[0][2]
    pad_col_x = [max(data_len_x) - x for x in data_len_x]
    pad_col_y = [max(data_len_y) - y for y in data_len_y]
    data_x = []
    data_y = []
    for x_cols, y_cols, data in zip(pad_col_x, pad_col_y, batch):
        X = np.array(data[0])
        X = np.pad(X, (0, x_cols), 'constant')
        Y = np.array(data[1])
        Y = np.pad(Y, (0, y_cols), 'constant')

        data_x.append(X)
        data_y.append(Y)

    X = torch.LongTensor(data_x)
    Y = torch.LongTensor(data_y)
    return X, Y, data_len_x, n_words

def evaluateRandomly(encoder, decoder, train_loader_vaild,index2word):
    for i, trainset in enumerate(train_loader_vaild):
        x, y, len_x, n_words = trainset
        decoded_words = evaluate(encoder, decoder, x, y, len_x, index2word)
        print('>', ' '.join([index2word[i] for i in x.item()]))
        print('=', ' '.join([index2word[i] for i in y.item()]))

        output_sentence = ' '.join(decoded_words)
        print('<', output_sentence)
        print('')


if __name__ == '__main__':
    vis = Visualizer(env='seq2seq',port=2333)
    print('---构建模型---')
    n_words = np.load('./n_words.npy').item()
    index2word = np.load('./index2word.npy').item()

    enc = encoderModel(n_words)
    dec = ArrdecoderModel(n_words)
    print(enc)
    print(dec)
    Datas = ShipDataset(Train=True,target='train.zh', src='train.en',dir='./')
    Datas_valid = ShipDataset(Train=False, target='train.zh', src='train.en', dir='./')
    train_loader = data.DataLoader(Datas, batch_size=3, num_workers=0, shuffle=False, collate_fn=my_collate)
    train_loader_vaild = data.DataLoader(Datas_valid, batch_size=1, num_workers=0, shuffle=False, collate_fn=my_collate)
    print('---训练---')
    trainIters(enc,dec,train_loader,vis=vis,epoch=5)
    evaluateRandomly(enc, dec, train_loader_vaild, index2word)


