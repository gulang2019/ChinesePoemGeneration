# ChinesePoetryGeneration

## 整体逻辑


word_dict.py 生成词典

word_vec2.py 生成词向量，依赖：word_dict.py

poems.py 由corpus生成古诗 

rank_words. 用共现矩阵和pagerank方法挑选关键词，依赖：word_dict, poems

data_utils 生成generate data, plan data, 依赖：rank_words, poem, word_dict

generator_2021 生成古诗， 依赖：word_dict, word_vec2, data_utils(get_batch)

改进的地方一个是字向量，一个是beamer筛选，一个是pagerank，一个是模型

"# ChinesePoemGeneration" 



models文件夹

存储模型，需要建立5个子文件夹，pair, sentence1, sentence2, sentence3, sentence4

result文件夹 存储实验结果(loss, attention graph, poem) 需要建立5个子文件夹，pair, sentence1, sentence2, sentence3, sentence4

data 文件夹

存储预处理后的输入文件， 需要建立5个子文件夹，pair, sentence1, sentence2, sentence3, sentence4

embedding_128.npy 存储128维度的词向量，这个字向量是网上找的

word_dict_128.txt 存储字典

sxhy_dict.txt 存储诗学含英中的词

poem_128存储源古诗


