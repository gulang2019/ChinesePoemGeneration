from plan import Planner
from generator_2021 import Generator
from typing import Set, List
import os
from word_vec2 import word2Vec
from word_dict import wordDict
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# For testing purpose.
if __name__ == '__main__':
    word_dict = wordDict()
    word2vec = word2Vec()
    generator = Generator()
    i= 0
    while True:
        i += 1
        hint : str = input("Type first sentence: ")
        print(hint)
        keywords = [i for i in list(hint) if word_dict.word2int(i) != -1]
        keyword = keywords[0]
        keywords = keywords[1:] + word2vec.similar_word_(keywords[0], 5 - len(keywords))
        keywords = [keyword] + [i for i in keywords if i != keyword]
        print("Keywords: ", keywords)
        keywords = [' '.join(list(i)) for i in keywords]
        poem : List[str] = generator.generate_by_multiple_models(keywords, 0,'./result/demo','',f'{i}')
        output = ''.join(poem.split()).strip('^').replace('$','\n')
        print("Poem: \n", output)
        with open("./result/demo/result.txt", 'a', encoding='utf-8') as f:
            f.write("Input: " + hint + '\n')
            f.write("Keywords: " + str(keywords) + '\n')
            f.write("Poem: \n")
            f.write(str(output))
            f.write('\n\n')
