# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:27:05 2017

@author: E601
"""
import tensorflow as tf
class BLSTM():
    def __init__(self,word_nums,output_size,embedding_dim, lstm_size, word_embeddings, 
                 init_learning_rate,decay_steps,learning_rate_decay):
        # Basic parameters
        self.word_nums=word_nums
        self.output_size = output_size
        self.embedding_dim=embedding_dim
        self.lstm_size = lstm_size
        self.init_learning_rate=init_learning_rate
        self.decay_steps=decay_steps
        self.learning_rate_decay=learning_rate_decay
        # Embeddings
        self.word_lookups = tf.constant(word_embeddings,dtype=tf.float32)
        self.cap_lookups = tf.constant([[1,0,0],[0,1,0],[0,0,1]], dtype=tf.float32)
     
        # Input
        # Shape: 1x sequence_length
        self.input_words = tf.placeholder(tf.int32, shape=(None,None)) 
        self.input_caps = tf.placeholder(tf.int32, shape=(None,None))
        self.dropout = tf.placeholder(dtype=tf.float32,name="Dropout")
        # Gold output
        self.true_tag_nums = tf.placeholder(tf.int32, shape=(None,None)) # Shape: batch_size x sequence_length
        
        #shape为:1*句子的长度*(embedding_dim+3)
        embedding =self.embedding_layer()
        self.lstm_inputs = tf.nn.dropout(embedding, self.dropout)
        self.logits=self.biLSTM_layer()
        self.loss_layer()
    def embedding_layer(self, name=None):
        """
        :param char_inputs: char feature
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, seq_len, embedding_dim+3], 
        """
        with tf.variable_scope("word_embedding" if not name else name):
             #embedding_lookup(params, ids)其实就是按照ids顺序返回params中的第ids行。
             self.input_word_embeddings=tf.nn.embedding_lookup(self.word_lookups, self.input_words)
        with tf.variable_scope("cap_embedding"):
             #self.input_word_embeddings的shape为[1,sequence_length,embedding_dim]
             self.input_cap_embeddings = tf.nn.embedding_lookup(self.cap_lookups, self.input_caps)

        self.cap_weights = tf.Variable(tf.random_uniform((3,self.embedding_dim),minval=-0.1,
                                                         maxval=0.1),dtype=tf.float32)
        self.words_with_caps = tf.reshape(self.input_word_embeddings, (-1,self.embedding_dim)) + \
        tf.matmul(tf.reshape(self.input_cap_embeddings, (-1,3)), self.cap_weights)
#        lstm_input为字符编号和大小写编号*大小写的权重求和得到，然后将其变成input_word_embeddings
#        的形状
        embed = tf.reshape(self.words_with_caps, tf.shape(self.input_word_embeddings))
        return embed
    def biLSTM_layer(self, name=None):
        # Bidirectional RNN (LSTM) layer
        forward_cell  = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.lstm_size), 
                                                           state_keep_prob=self.dropout, 
                                                           input_keep_prob=self.dropout)
        backward_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.lstm_size),
                                                           state_keep_prob=self.dropout, 
                                                           input_keep_prob=self.dropout)
        (self.output_fw, self.output_bw), _ = tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                        backward_cell, self.lstm_inputs, dtype=tf.float32)
        self.flat_output_fw = tf.reshape(self.output_fw, (-1, self.lstm_size))
        self.flat_output_bw = tf.reshape(self.output_bw, (-1, self.lstm_size))


        # Output layer
        self.out_fw_weight = tf.Variable(tf.random_uniform((self.lstm_size,self.output_size),minval=-0.1,
                                                           maxval=0.1),dtype=tf.float32)
        self.out_bw_weight = tf.Variable(tf.random_uniform((self.lstm_size,self.output_size),minval=-0.1,
                                                           maxval=0.1), dtype=tf.float32)
        self.out_bias = tf.Variable(tf.random_uniform((self.output_size,), minval=-0.1, maxval=0.1),
                                                                                 dtype=tf.float32)

        self.flat_output = tf.matmul(self.flat_output_fw, self.out_fw_weight) + tf.matmul(self.flat_output_bw, self.out_bw_weight) + self.out_bias
        return tf.reshape(self.flat_output, (-1, tf.shape(self.input_words)[1], self.output_size))

    def loss_layer(self, name=None):
        """
        calculate crf loss
        :param project_logits: [batch_size, max_seq_len, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("softmax_loss"  if not name else name):
             # Optimization & training
             self.global_step = tf.Variable(0, trainable=False)
             self.learning_rate = tf.train.exponential_decay(self.init_learning_rate,  self.global_step, self.decay_steps,
                                                            self.learning_rate_decay, staircase=True)
             self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.true_tag_nums))
             self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, global_step= self.global_step)
        with tf.variable_scope("softmax_pred_op"  if not name else name):
             labels_softmax = tf.argmax(self.logits, axis=-1)
             predicted_tag_numbers= tf.cast(labels_softmax, tf.int32)
        with tf.name_scope("accuracy"):
             labels_softmax = tf.argmax(self.logits, axis=-1)
             predicted_tag_numbers= tf.cast(labels_softmax, tf.int32)
             correct_prediction = tf.equal(predicted_tag_numbers, self.true_tag_nums)
             self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   
