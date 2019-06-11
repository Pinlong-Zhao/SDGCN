# -*- coding: utf-8 -*-
# mul means aspect2sentence attention and sentence2aspect attention
import tensorflow as tf
import numpy as np
from models.nn_layer import dynamic_rnn, softmax_layer, bi_dynamic_rnn, reduce_mean_with_len,WXA_Relu
from models.att_layer import dot_produce_attention_layer, bilinear_attention_layer, mlp_attention_layer, Mlp_attention_layer

class CAtt(object):
    def __init__(self, sequence_length, target_sequence_length,targets_num_max, num_classes, word_embedding, l2_reg_lambda=0.0,
                 num_hidden=100):

        # PLACEHOLDERS
        rand_base = 0.01
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")  # X - The Data
        self.input_target = tf.placeholder(tf.int32, [None, target_sequence_length], name="input_x")  # The target
        self.input_targets_all = tf.placeholder(tf.int32, [None,targets_num_max, target_sequence_length], name="input_x")  #All the targets

        self.sen_len = tf.placeholder(tf.int32, None, name='sen_len')#lens of sentence
        self.target_len = tf.placeholder(tf.int32, None, name='target_len')#lens of target
        with tf.name_scope('targets_all_len'):
            self.targets_all_len_a = tf.placeholder(tf.int32, [None,targets_num_max],name="targets_all_len")
            batch_size = tf.shape(self.input_x)[0]
            self.targets_all_len = []
            for i in range(targets_num_max):
                targets_i_len = tf.slice(self.targets_all_len_a, [0, i], [batch_size, 1])
                self.targets_all_len.append(tf.squeeze(targets_i_len))              #lens of every target
        self.targets_num = tf.placeholder(tf.int32, None, name='targets_num')     #The number os targets
        self.relate_cross = tf.placeholder(tf.float32, [None,targets_num_max, targets_num_max], name='relate_cross')  #the relation between targets
        self.relate_self = tf.placeholder(tf.float32, [None, targets_num_max, targets_num_max], name='relate_self')
        self.target_which = tf.placeholder(tf.float32, [None, targets_num_max ], name='which_one_target')
        self.target_position = tf.placeholder(tf.float32, [None, sequence_length], name='target_position')
        with tf.name_scope('targets_all_position'):
            self.targets_all_position_a = tf.placeholder(tf.float32, [None,targets_num_max,sequence_length],name="targets_all_position")
            self.targets_all_position = []
            for i in range(targets_num_max):
                targets_i_len = self.targets_all_position_a[:, i,:]
                self.targets_all_position.append(tf.squeeze(targets_i_len))
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")  # Dropout

        l2_loss = tf.constant(0.0)  # Keeping track of l2 regularization loss

        # 1. EMBEDDING LAYER ################################################################
        with tf.name_scope("embedding"):
            self.word_embedding = tf.constant(word_embedding, name='word_embedding')
        # Embedding for the context
        with tf.name_scope("embedded_sen"):
            self.embedded_sen = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
            # self.embedded_expanded = tf.expand_dims(self.embedded, -1)
            self.embedded_sen = tf.cast(self.embedded_sen, tf.float32)      #(?,78,300)
            self.embedded_sen = tf.nn.dropout(self.embedded_sen, keep_prob=self.dropout_keep_prob)
            embedding_size = word_embedding.shape[1]
            print('embedding_size {}'.format(embedding_size))
            num_hidden = embedding_size
        # Embedding for the target
        with tf.name_scope("embedding_target"):
            self.embedded_target = tf.nn.embedding_lookup(self.word_embedding, self.input_target)
            self.embedded_target = tf.cast(self.embedded_target, tf.float32)    #(?,21,300)
            self.embedded_target = tf.nn.dropout(self.embedded_target, keep_prob=self.dropout_keep_prob)



        #2. LSTM LAYER ######################################################################
        # Bi-LSTM for the context
        with tf.name_scope("Bi-LSTM_sentence"):
            cell = tf.nn.rnn_cell.LSTMCell
            self.LSTM_Hiddens_sen = bi_dynamic_rnn(cell, self.embedded_sen, num_hidden, self.sen_len,
                                           sequence_length, 'bi-lstm-sentence' ,'all',
                                           dropout = True, dropout_prob= self.dropout_keep_prob) #(?,78,600)
            pool_sen = reduce_mean_with_len(self.LSTM_Hiddens_sen, self.sen_len)
        # Bi-LSTM for the target
        with tf.name_scope("Bi-LSTM_target"):
            cell = tf.nn.rnn_cell.LSTMCell
            self.LSTM_Hiddens_target = bi_dynamic_rnn(cell, self.embedded_target, num_hidden, self.target_len,
                                            target_sequence_length, 'bi-lstm-target', 'all',
                                            dropout=True, dropout_prob=self.dropout_keep_prob)  # (?,21,600)
            pool_target = reduce_mean_with_len(self.LSTM_Hiddens_target, self.sen_len)



        # 3. Attention LAYER ######################################################################
        #target to sentence attention

        with tf.name_scope("Attention-sentence2target"):
            self.att_t = bilinear_attention_layer(self.LSTM_Hiddens_target, pool_sen, self.target_len,2 * num_hidden ,l2_reg_lambda,
                                             random_base = rand_base, layer_id = 'tar')   #(?,1,78)
            self.outputs_t = tf.squeeze(tf.matmul(self.att_t, self.LSTM_Hiddens_target), axis=1)     #(?,1,78)* (?,78,600)----->(?,600)


        with tf.name_scope("Attention-target2sentence"):
            # position
            target_position = tf.expand_dims(self.target_position, 2)  # (?,78,1)
            LSTM_Hiddens_sen_p = tf.multiply(self.LSTM_Hiddens_sen, target_position)

            self.att_s = bilinear_attention_layer(LSTM_Hiddens_sen_p, self.outputs_t, self.sen_len,2 * num_hidden ,l2_reg_lambda,
                                             random_base = rand_base, layer_id = 'sen')   #(?,1,78)
            self.outputs_s = tf.squeeze(tf.matmul(self.att_s, self.LSTM_Hiddens_sen), axis=1)     #(?,1,78)* (?,78,600)----->(?,600)


        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal([num_hidden * 2, num_classes]))
            b = tf.Variable(tf.random_normal([num_classes]))
            self.scores = tf.nn.xw_plus_b(self.outputs_s, W,b, name="scores")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.true_y = tf.argmax(self.input_y, 1, name="true_y")
            self.softmax = tf.nn.softmax(self.scores, name="softmax")


        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss") + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(self.predictions,self.true_y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"),name="accuracy")
        print ("LOADED CATT!")


