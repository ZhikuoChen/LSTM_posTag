# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 23:18:23 2017

@author: E601
"""
'''
#基本名词短语(base NP)指简单的，非嵌套的名词短语，不含有其他子短语
词性标注(Part-of-Speech tagging)
B-NP基本名词短语;CC(并列连词);CD(cardinadal number基数);DT(定冠词);EX(存在性，如there);FW(foreign word);
IN(介词或从属连词);JJ(adj);JJR(形容词比较级);JJS(形容词最高级);LS(list item marker);MD(modal语气词)
NN(名词);NNS(名词复数);NNP(专有名词单数);NNPS(专有名词复数);NP(名词短语);PDT(predeterminer);
POS(possessive ending所有格结尾);PP(介词短语);PRP(人称代词); PRP$(人称代词所有格);RB(副词);RBR(副词比较级);
RBS(副词最高级);RP(助动词);SYM(符号);TO(to);UH(interjection感叹词);VB(动词基本形式);VBD(动词过去式)
VBG(动名词或现在分词);VBN(动词过去分词);VBP(非第三人称单数);VBZ(第三人称单数);VP(动词短语);
WDT(wh做定语);WP(wh-代词);WP$(wh-代词的所有格形式);WRB(wh的副词形式)
'''
import time,argparse
import tensorflow as tf
from blstm_posTag import BLSTM
import os
import numpy as np
from load_data import load_data,load_embeddings,load_tagset

#hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM for POS Tag task')
parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
parser.add_argument('--num_epochs', type=int, default=50, help='#epoch of training')
parser.add_argument('--train_words_file', type=str, default="data/train_words.txt", help='train data source')
parser.add_argument('--train_tags_file', type=str, default="data/train_tags.txt", help='test data source')
parser.add_argument('--dev_words_file', type=str, default="data/dev_words.txt", help='train data source')
parser.add_argument('--dev_tags_file', type=str, default="data/dev_tags.txt", help='test data source')
parser.add_argument('--tagset_file', type=str, default="data/tagset.txt", help='test data source')
parser.add_argument('--embeddings_file', type=str, default='data/glove.6B.200d.txt', help='word embedding data source')
parser.add_argument('--embedding_dim', type=int, default=200, help='random init char embedding_dim')
parser.add_argument('--lstm_dim', type=int, default=64, help='Num of hidden units in LSTM')

parser.add_argument('--optimizer', type=str, default='Adagrad', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--init_learning_rate', type=float, default=0.003, help='learning rate')
parser.add_argument('--learning_rate_decay', type=float, default=0.95, help='Learning rate decay rate')
parser.add_argument('--decay_steps', type=int, default=10802, help='decay steps')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument("--checkpoint_every",type=int,default=1000, help="Save model after this many steps (defult: 50)")
args = parser.parse_args()

#检查点目录。 
checkpoint_dir  = ".\data\checkpoints"
##如果该文件不存在，则重新创建
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# Load data
word_to_nums, num_to_words, word_embeddings = load_embeddings(args.embeddings_file, args.embedding_dim)
word_embeddings=np.array(word_embeddings)
tag_to_nums, num_to_tags = load_tagset(args.tagset_file)
#{'#': 0, '$': 1, "''": 2, ',': 3, '-LRB-': 4, '-NONE-': 5, '-RRB-': 6, '.': 7, ':': 8, 'CC': 9,
# 'CD': 10, 'DT': 11, 'EX': 12, 'FW': 13, 'IN': 14, 'JJ': 15, 'JJR': 16, 'JJS': 17, 'LS': 18,
# 'MD': 19, 'NN': 20, 'NNP': 21, 'NNPS': 22, 'NNS': 23, 'PDT': 24, 'POS': 25, 'PRP': 26, 
# 'PRP$': 27, 'RB': 28, 'RBR': 29, 'RBS': 30, 'RP': 31, 'SYM': 32, 'TO': 33, 'UH': 34, 'VB': 35,
# 'VBD': 36, 'VBG': 37, 'VBN': 38, 'VBP': 39, 'VBZ': 40, 'WDT': 41, 'WP': 42, 'WP$': 43, 'WRB': 44,
# '``': 45}

train_sents,dev_sents =load_data(args.train_words_file, args.train_tags_file,
                                 args.dev_words_file,args.dev_tags_file,word_to_nums,
                                 tag_to_nums)

#word_embeddings为已训练好的每个词的词向量组成的二维数组，每一行为一个词的词向量
#每个词可通过通过字典word_to_nums，找到自己对应的200维词向量
#train_sents为所有训练集组成的列表，由每一行的set组成(word_numbers, caps, tag_nums)
#word_numbers为一行句子中，每个单词对应的编号组成的列表
#caps为一行句子中，每个单词的首字母大写状态对应的编号组成的列表
#tag_nums为一行句子中，每个单词的tag对应的编号组成的列表

with tf.Session() as sess:
     lstm = BLSTM(len(word_to_nums),len(tag_to_nums),args.embedding_dim,args.lstm_dim,
                  word_embeddings,args.init_learning_rate,args.decay_steps,
                  args.learning_rate_decay)
     saver = tf.train.Saver(tf.global_variables())
     sess.run(tf.global_variables_initializer())
     if args.mode=='train':
        print("Start training")
        for epoch in range(args.num_epochs):
            np.random.shuffle(train_sents)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            total_loss=0
            #因为此程序没有进行所有句子的等长处理，所以一次只能训练一句话。
            for (word_nums, caps, tag_nums) in train_sents:
                #对word_nums，caps，tag_nums加一维度，使其由一维变成两维
                feed_dict={lstm.input_words: [word_nums], lstm.input_caps: [caps],
                                      lstm.true_tag_nums: [tag_nums],lstm.dropout:args.dropout}
                loss,_,step=sess.run([lstm.loss,lstm.train_op,lstm.global_step],feed_dict)
                total_loss+=loss

            # 每checkpoint_every次执行一次保存模型
            Path=saver.save(sess, os.path.join(checkpoint_dir,'model.ckpt'), global_step=epoch+1)   # 定义模型保存路径
            print("Save model checkpoint to ",Path)
            print("[%s]:epoch:%d step:%d loss:%f" %(timestamp,epoch+1, step+1, total_loss))
     elif args.mode=='test':
          print("Start testing")
            #初始化所有变量
          ckpt = tf.train.latest_checkpoint(checkpoint_dir)
          if ckpt: 
             saver.restore(sess, ckpt)
             print("Model restored.")
          accuracy_list=[]
          for (word_nums, caps, tag_nums) in dev_sents:
              feed_dict={lstm.input_words: [word_nums], lstm.input_caps: [caps],lstm.true_tag_nums: [tag_nums],lstm.dropout:1.0}
              accuracy,_=sess.run([lstm.accuracy,lstm.train_op],feed_dict=feed_dict)
              accuracy_list.append(accuracy)
          test_acc=sum(accuracy_list)/len(accuracy_list)
          print("After traing,accuracy on test set is:", test_acc)
