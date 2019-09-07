from torch.utils import data
import os
import re
import torch.nn as nn
import torch
import jieba
import linecache
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

'''
    读取翻译数据
    这里我需要处理的非常简单，去掉一些特殊的标点符号。
    中文的jieba分词
    将单词的索引作为输出，统计并打印数据集大小和随机例子
'''

class ShipDataset(data.Dataset):
    def __init__(self,Train,target,src,dir='../'):
        super(ShipDataset, self).__init__()
        self.dir = dir   #数据目录
        self.trg = target
        self.src = src
        self.word2index = {'SOS':0,'EOS':1}
        self.index2word = {0:'SOS',1:'EOS'}
        self.word2count = {}
        self.n_words = 2
        self.rols = 0  #文件中的行数，代表多少条数据
        self.vaild_rols = 0
        self.Train = Train
        self.valid_en = 'valid.en.sgm'
        self.valid_zn = 'valid.zh.sgm'
        if os.path.exists('word2index.npy'):
            self.word2index = np.load('word2index.npy', allow_pickle=True).item()
            self.index2word = np.load('index2word.npy', allow_pickle=True).item()
            self.rols = np.load('rols.npy',allow_pickle=True).item()
            self.n_words = np.load('n_words.npy', allow_pickle=True).item()
            self.vaild_rols = np.load('vaild_rols.npy', allow_pickle=True).item()

        else:

            self.initKeyWord(src)
            self.initKeyWord(target)
            self.initVaildWord(self.valid_zn)
            self.initVaildWord(self.valid_en)

            np.save('word2index.npy', self.word2index)
            np.save('index2word.npy', self.index2word)
            np.save('rols.npy',self.rols)
            np.save('n_words.npy', self.n_words)
            np.save('vaild_rols.npy', self.vaild_rols)


    def __getitem__(self, item):
        item +=1
        if self.Train == True:
            trg = linecache.getline(self.dir + self.trg, item)
            src = linecache.getline(self.dir + self.src, item)

            x = self.SegWord(self.src, src)
            y = self.SegWord(self.trg, trg)


        else:
            tree = ET.parse(self.dir+self.valid_en)
            root = tree.getroot()
            for seg in root.iter('seg'):
                if seg.get('id') == str(item):
                    x = self.SegWord(self.valid_en, seg.text)
                    break
            tree = ET.parse(self.dir + self.valid_zn)
            root = tree.getroot()
            for seg in root.iter('seg'):
                if seg.get('id') == str(item):
                    y = self.SegWord(self.valid_zn, seg.text)
                    break
        # 将词的索引取代词，封装成Longtensor
        X = [self.word2index[i] for i in x]
        Y = [self.word2index[i] for i in y]
        Y.insert(0,0)  #添加SOS
        Y.append(1)     #添加EOS
        return X,Y,self.n_words     #返回这个主要是为了训练的embeding参数

    def __len__(self):
        #返回数据的数量
        if self.Train == True:
            return self.rols-1
        else:
            return self.vaild_rols

    def initKeyWord(self,filename):

        with open(self.dir+filename, 'r', encoding='utf-8') as f_en:
            for line in tqdm(f_en,desc='构建字库', leave='False'):
                self.rols += 1   #记录行数

                s= self.SegWord(filename,line)

                for word in s:
                    if word not in self.word2index:
                        self.word2index[word] = self.n_words
                        self.index2word[self.n_words] = word
                        self.word2count[word] = 1
                        self.n_words += 1
                    else:
                        self.word2count[word] += 1
    def initVaildWord(self,filename):
        tree = ET.parse(self.dir+filename)
        root = tree.getroot()
        for seg in tqdm(root.iter('seg')):
            s = seg.text
            s = self.SegWord(filename, s)
            for word in s:
                if word not in self.word2index:
                    self.word2index[word] = self.n_words
                    self.index2word[self.n_words] = word
                    self.word2count[word] = 1
                    self.vaild_rols += 1
                else:
                    self.word2count[word] += 1

    def SegWord(self,filename,line):
        line = line.lower().strip()  # 全部小写，去除尾部回车符
        s = re.sub(r'([-。，？“”()（）.!?])', r'', line)  # 去除特殊符号
        suffix = filename.split('.')[1]  # 获取文件后缀
        if suffix == 'zh' :  # 需要jieba分词
            s = ' '.join(jieba.cut(s))

        s = s.split(' ')  # 按照空格将词分开
        s = list(filter(None, s))  #去掉空字符
        return s
