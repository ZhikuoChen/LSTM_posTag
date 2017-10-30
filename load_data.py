import numpy as np
def load_sents(words_file, tags_file):
    '''Load words and tags from files'''
    words = list()
    tags = list()
    with open(words_file,"r",encoding='utf8') as f:
         with open(tags_file,"r",encoding='utf8') as g:
              for line in f:
                  line = line.strip().split('\t')
                  words.append(line)
              for line in g:
                  line = line.strip().split('\t')
                  tags.append(line)
                
    assert len(words) == len(tags)
    #train_sents为一列表，列表内容为每一句话的单词构成的列表和该句话每个单词的词性构成的列表合成的
    #set构成的
    #如(['In', 'an', 'Oct.', '19', 'review', 'of', '``', 'The', 'Misanthrope',
        #"''", 'at', 'Chicago', "'s", 'Goodman', 'Theatre', '-LRB-', 
        #'``', 'Revitalized', 'Classics', 'Take', 'the', 'Stage', 'in', 'Windy', 
        #'City', ',', "''", 'Leisure', '&', 'Arts', '-RRB-', ',', 'the', 'role',
        #'of', 'Celimene', ',', 'played', '*', 'by', 'Kim', 'Cattrall', ',', 
        #'was', 'mistakenly', 'attributed', '*-2', 'to', 'Christina', 'Haag', '.'],
    #['IN', 'DT', 'NNP', 'CD', 'NN', 'IN', '``', 'DT', 'NN', "''", 'IN', 'NNP', 
    #'POS', 'NNP', 'NNP', '-LRB-', '``', 'VBN', 'NNS', 'VBP', 'DT', 'NN', 'IN', 
    #'NNP', 'NNP', ',', "''", 'NN', 'CC', 'NNS', '-RRB-', ',', 'DT', 'NN', 'IN',
    #'NNP', ',', 'VBN', '-NONE-', 'IN', 'NNP', 'NNP', ',', 'VBD', 'RB', 'VBN', 
    #'-NONE-', 'TO', 'NNP', 'NNP', '.'])
    sents = [(words[i], tags[i]) for i in range(len(words))]
    return sents

def load_embeddings(embeddings_file, embeddings_dim):
    '''Load word embeddings of the specified dimension from a file'''
    word_to_nums = dict()
    num_to_words = dict()
    word_embeddings = list()

    with open(embeddings_file,"r",encoding='utf8') as f:
        i = 0
        for line in f:
            line = line.split(' ')
            word = line[0]
            vals = list(map(float, line[1:embeddings_dim+1]))
            #word_to_nums为一字典，键为单词，值为该单词对应的编号，如the:0
            word_to_nums[word] = i
            #num_to_word为一字典，键为该单词对应的编号，值为单词，如0:the
            num_to_words[i] = word
            #word_embeddings为二维列表，对应每个单词训练好的词向量
            word_embeddings.append(vals)
            i += 1
    #并将'TRACE'，'NUMBER','UNK'等特殊词汇加入词汇表中，并随机产生这些词的词向量
    # Add traces etc.
    word_to_nums['TRACE'] = i
    num_to_words[i] = 'TRACE'
    word_embeddings.append([np.random.uniform(low=-0.1, high=0.1) for _ in range(embeddings_dim)])
    i += 1 

    word_to_nums['NUMBER'] = i
    num_to_words[i] = 'NUMBER'
    word_embeddings.append([np.random.uniform(low=-0.1, high=0.1) for _ in range(embeddings_dim)])
    i += 1

    word_to_nums['UNK'] = i
    num_to_words[i] = 'UNK'
    word_embeddings.append([np.random.uniform(low=-0.1, high=0.1) for _ in range(embeddings_dim)])
    i += 1

    return word_to_nums, num_to_words, word_embeddings

def load_tagset(tagset_file):
    i = 0
    tag_to_nums = dict()
    num_to_tags = dict()
    with open(tagset_file,"r",encoding='utf8') as f:
        for line in f:
            tag = line.strip()
            #tag_to_nums为字典，键为词性,值为该词性对应的编号。
            #{'#': 0, '$': 1, "''": 2, ',': 3, '-LRB-': 4, '-NONE-': 5, '-RRB-': 6, 
            #'.': 7, ':': 8, 'CC': 9, 'CD': 10, 'DT': 11, 'EX': 12, 'FW': 13, 
            #'IN': 14, 'JJ': 15, 'JJR': 16, 'JJS': 17, 'LS': 18, 'MD': 19, 'NN': 20,
            #'NNP': 21, 'NNPS': 22, 'NNS': 23, 'PDT': 24, 'POS': 25, 'PRP': 26, 
            #'PRP$': 27, 'RB': 28, 'RBR': 29, 'RBS': 30, 'RP': 31, 'SYM': 32,
            #'TO': 33, 'UH': 34, 'VB': 35, 'VBD': 36, 'VBG': 37, 'VBN': 38, 
            #'VBP': 39, 'VBZ': 40, 'WDT': 41, 'WP': 42, 'WP$': 43, 'WRB': 44, 
            #'``': 45}
            tag_to_nums[tag] = i
            #num_to_tags为字典，键为该词性对应的编号,值为词性。
            num_to_tags[i] = tag
            i += 1

    return tag_to_nums, num_to_tags

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def capitalization(string):
    if string.isupper():
        return 2
    elif string[0].isupper():
        return 1
    else:
        return 0

def normalize_sent(sent):
    sent = [word.lower() for word in sent]
    normalized_sent = list()
    for word in sent:
        if word == "-lrb-" or word == "-lcb-":
            word = "("
        elif word == "-rrb-" or word == "-rcb-":
             word = ")"
        elif word.startswith("*") or word == "0":
            word = "TRACE"
        elif is_number(word):
            word = "NUMBER"
        normalized_sent.append(word)
        
    assert len(normalized_sent) == len(sent)
    #normalized_sent为一句话经过一些"标准化处理"后的词组成的列表
    return normalized_sent

def sent_to_numbers(train_sents,dev_sents, word_to_nums, tag_to_nums):
    #训练集:
    trainSents=[]
    for train_sent in train_sents:
        train_words, train_tags = train_sent
        #word_nums为一句话中每个词的编号构成的列表
        train_word_numbers = [word_to_nums[word] if word in word_to_nums else word_to_nums["UNK"] for word in normalize_sent(train_words)]
        #caps为一句话中每个词的大写状态编号构成的列表，如该词全为大写，则该词的编号为2，
        #该词首字母大写，则该词的编号为1，该词都为小写，则该词的编号为0，
        train_caps = list(map(capitalization, train_words))
        #tag_numbers为一句话中每个词的词性编号构成的列表
        train_tag_numbers = [tag_to_nums[tag] for tag in train_tags]
        assert len(train_word_numbers) == len(train_caps) == len(train_tag_numbers)
        trainSents.append((train_word_numbers, train_caps, train_tag_numbers))
    devSents=[]
    for dev_sent in dev_sents:
        #测试集:
        dev_words, dev_tags = dev_sent
        #word_nums为一句话中每个词的编号构成的列表
        dev_word_numbers = [word_to_nums[word] if word in word_to_nums else word_to_nums["UNK"] for word in normalize_sent(dev_words)]
        #caps为一句话中每个词的大写状态编号构成的列表，如该词全为大写，则该词的编号为2，
        #该词首字母大写，则该词的编号为1，该词都为小写，则该词的编号为0，
        dev_caps = list(map(capitalization, dev_words))
        #tag_numbers为一句话中每个词的词性编号构成的列表
        dev_tag_numbers = [tag_to_nums[tag] for tag in dev_tags]
        assert len(dev_word_numbers) == len(dev_caps) == len(dev_tag_numbers)
        devSents.append((dev_word_numbers, dev_caps, dev_tag_numbers))
    return trainSents,devSents
def load_data(train_words_file, train_tags_file,dev_words_file, dev_tags_file,
                                                    word_to_nums, tag_to_nums):
    #train_sents为一列表，列表内容为每一句话的单词构成的列表和该句话每个单词的词性构成的列表合成的
    #set构成的
    train_sents = load_sents(train_words_file, train_tags_file)
    dev_sents = load_sents(dev_words_file, dev_tags_file)
    trainSents,devSents=sent_to_numbers(train_sents, dev_sents,word_to_nums, tag_to_nums)
    return trainSents,devSents
