# seq2seq


![image](https://github.com/liAoI/RNN-pytorch--/blob/master/images_result/2019-09-07%2009-17-24seq2seq.png)

这是工程结构目录，里面的数据集可以从以下百度云获取到：

链接：https://pan.baidu.com/s/12pg77vTf87jEflhvLtzlhA 
提取码：hk7j 

训练的loss结果图如下：

![image](https://github.com/liAoI/RNN-pytorch--/blob/master/images_result/seq2seqforloss.png)

在训练中发现，程序可能会很吃内存，或者是哪个地方需要优化，总是训练不到一个epoch，就被Kill掉。有时间我会更新一下。还没有跑出验证结果就被Kill掉，也很无奈！

被Kill的原因找到了，主要是服务器别人也在用，用的人多了就自然被杀了。另外一个就是这个训练数据太大了，足足有近千万行翻译数据，用最简单的seq2seq网络跑两天都跑不完100个epoch。

下面的结果让人很失望，我有时间看看是代码哪里出了问题，这里记录一下结果

> do you think we look young enough to blend in at a high school

= SOS 你们 觉得 我们 看起来 够 年轻 溜进 高中 吗 EOS

< SOS 一 小时 前 我 的 的 的 的 的 的

> hi, honey i guess you're really tied up in meetings

= SOS 嗨 亲爱 的 你 现在 肯定 忙 着 开会 呢 EOS

< SOS 一 小时 前 我 就 我 的 的 的 EOS EOS

> because you want to start a family before you hit the nursing home

= SOS 因为 你 想 在 进 养老院 前 娶妻生子 EOS

< SOS 一 小时 前 我 的 的 的 的 的

> she's got to have me in her sight like 24 hours a day

= SOS 我 就 一天 24 小时 都 得 在 她 眼皮子 底下 EOS

< SOS 一 小时 前 我 就 的 的 的 的 EOS EOS EOS

> find a safety chain or something to keep these lights in place

= SOS 找条 牢靠 的 链子 或者 别的 什么 固定 住 这些 灯 EOS

< SOS 一 小时 前 我 的 的 的 的 的 EOS EOS SOS

> so that no parent has to go through what i've known

= SOS 为了 不让 别的 父母 经历 我 的 遭遇 EOS

< SOS 一 小时 前 我 就 我 的 的 EOS
