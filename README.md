# text-classification-with-python
用朴素贝叶斯算法和结巴分词实现新闻主题分类，Python代码
folder_path = '/Users/apple/Documents/七月在线/NLP/第2课/Lecture_2/Naive-Bayes-Text-Classifier/Database/SogouC/Sample'
stopwords_file = '/Users/apple/Documents/七月在线/NLP/第2课/Lecture_2/Naive-Bayes-Text-Classifier/stopwords_cn.txt'
 

下载地址：链接:https://pan.baidu.com/s/1O5mW04PlulaCW5TUd93OUg  密码:ubkq

然后切换Python2.7，跑下面代码就可以进行自然语言入门了



# 1 数据采集

用Python爬虫从腾讯新闻中心 RSS 源采集下来

# 2 数据清洗

采用正则表达式 import re实现

# 3. 特征选择

采用 jieba 分词模块来对文本进行分词及提取有代表性的关键词作为特征，jieba 分词模块自带
的词库中包含着每个词的词频(TF)及反文档频率(IDF)，每个词的 TF 值，IDF 值均由原作者通过大量文本
训练统计出来的，所以具有一般性，使用该方法得到的关键词用人工标准来判断能反映出文本主题。当
使用 jieba 分词模块的提取关键词功能时，它会对在对文本进行分词的同时会利用每个词的 TF 值及 IDF
值计算出每个词的权重(Weight = TF*IDF)，然后根据权重大小对词进行排序，至于返回前多少歌词则由
用户设定。另外，在使用提取特征词功能的时候还能去除标点符号及对文本主题无意义的停用词。
根据 jieba 分词模块提取关键词的方法可知，它直接可以对单个文本提取关键词，利用这个特点，在
对待分类文本也作关键词提取处理，只保留当中有代表性的关键词，这样既能大大减少生成词向量的时
间又能提高分类准确率。



# 4. 朴素贝叶斯模型的训练

是利用已转换为词向量的训练文本计算出每类文本的先验概率
用函数MultinomialNB().fit()来进行训练



# 5 模型预测

预测函数predict函数对测试集进行预测

先对待分类文本进行关键词提取，每篇提取前 20 个权重最大的词，再转换成词向量，然后与模型训
练计算出来的先验概率 p(di   ci )一起计算出文本属于每一类文本的概率 p(ci   di )，然后比较大小，选择概
率最大的并判别文本属于哪个类别，输出类别标签。

最后用score函数生成预测结果
