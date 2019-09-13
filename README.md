# seq2seq


![image](https://github.com/liAoI/RNN-pytorch--/blob/master/images_result/2019-09-07%2009-17-24seq2seq.png)

这是工程结构目录，里面的数据集可以从以下百度云获取到：

链接：https://pan.baidu.com/s/12pg77vTf87jEflhvLtzlhA 
提取码：hk7j 

训练的loss结果图如下：

![image](https://github.com/liAoI/RNN-pytorch--/blob/master/images_result/seq2seqforloss.png)

在训练中发现，程序可能会很吃内存，或者是哪个地方需要优化，总是训练不到一个epoch，就被Kill掉。有时间我会更新一下。还没有跑出验证结果就被Kill掉，也很无奈！

被Kill的原因找到了，主要是服务器别人也在用，用的人多了就自然被杀了。另外一个就是这个训练数据太大了，足足有近千万行翻译数据，用最简单的seq2seq网络跑两天都跑不完100个epoch。
