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



| 关键字 | 扩展后的关键字           | 古诗                                                         | 数据集中诗                                            |
| ------ | ------------------------ | ------------------------------------------------------------ | ----------------------------------------------------- |
| 夜     | ['夜', '昼', '更', '日'] | **夜**半清阴雨，云高远度幽。<br/>兴来多旧事，因尔慰南楼。    | 夜雨沉清磬，霜林起暮鸦。<br/>莲台三品叶，佛果一时花。 |
| 花，鸟 | ['花', '鸿', '红', '春'] | **花**下红泉色，云迎乳**鹤**声。<br/>芙蓉自**红**舞，杨柳故流**春**。 | 花落吴山暮，鸿归楚树低。<br/>解鞍空伫立，无那子规啼。 |
| 水     | ['水', '又', '云', '流'] | **水**断瓜州驿，江连北固城。<br/>**云**云千里迥，日暮万家通。 | 水断瓜州驿，江连北固城。<br/>涨沙三十里，树杪乱山横。 |
| 鸟     | ['鸟', '禽', '飞', '时'] | 择木翔归**鸟**, 临渊聚戏鱼.<br/>霜华无所识, 风色满何无.      | 余霭浮孤鸟，残阳曳暮蝉。<br/>沧江虹贯月，谁问米家船。 |

