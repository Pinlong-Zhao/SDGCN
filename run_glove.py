# -*- coding: utf-8 -*-
'''
# @project : SDGCN
# @Author  : plzhao
# @Software: PyCharm
'''
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import data_helpers
from sklearn import metrics
from models.att import Att
from models.catt import CAtt
from models.att_gcn import Att_GCN
from models.catt_gcn import CAtt_GCN_L1,CAtt_GCN_L2,CAtt_GCN_L3,CAtt_GCN_L4,CAtt_GCN_L5,CAtt_GCN_L6,CAtt_GCN_L7,CAtt_GCN_L8
from models.catt_gcn_woP import CAtt_GCN_woP
# Parameters
# ==================================================
# "Restaurants" or "laptops"
use_data = "Restaurants"
# "Att"  "CAtt"  "Att_GCN"  "CAtt_GCN_L2"  "CAtt_GCN_woP"
use_model = "CAtt_GCN_L2"

datas = {"Restaurants_train": "data/data_res/Restaurants_Train.txt",
         "Restaurants_test": "data/data_res/Restaurants_Test.txt",
         "Restaurants_embedding": 'data/data_res/Restaurants_glove.42B.300d.txt',
         "Laptops_train": "data/data_lap/Laptops_Train.txt",
         "Laptops_test": "data/data_lap/Laptops_Test.txt",
         "Laptops_embedding": 'data/data_lap/Laptops_glove.42B.300d.txt'}
#Train model
tf.flags.DEFINE_string("which_relation", 'global', "use which relation.") #'adjacent','global','rule'
tf.flags.DEFINE_string("which_model", use_model, "Model isused.")

# Data loading params
tf.flags.DEFINE_string("which_data", use_data, "Data is used.")
tf.flags.DEFINE_string("train_file", datas[use_data+"_train"], "Train data source.")
tf.flags.DEFINE_string("test_file", datas[use_data+"_test"], "Test data source.")

#word embedding
tf.flags.DEFINE_string('embedding_file_path', datas[use_data+"_embedding"], 'embedding file')
tf.flags.DEFINE_integer('word_embedding_dim', 300, 'dimension of word embedding')

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning_rate (default: 1e-3)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 80, "Number of training epochs ")
tf.flags.DEFINE_integer("evaluate_every", 5, "Evaluate model on dev set after this many steps ")
tf.flags.DEFINE_integer("checkpoint_every", 5, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def preprocess():
    '''
    read from the text file.
    :return:    sen word id:[324,1413,1,41,43,0,0,0]
                sen len:[5]
                sen max len :[8]
                sen label:[0,0,1]
                target word id:[34,154,0,0]
                target len: [2]
                target max len: [4]
                targets word id :[[34,154,0,0],
                                  [34,14,12,56],
                                  [0,0,0,0]]
                targets num = 2
                targets len: [2,4,0]
                targets max num:[3]
                targets_relation_self = [[1,0,0],
                                         [0,1,0],
                                         [0.0.0]]
                targets_relation_cross = [[0,1,0],
                                          [1,0,0],
                                          [0.0.0]]
    '''
    # Data Preparation
    # ==================================================
    # Load data
    print("Loading data...")
    train_x_str,train_target_str, train_y = data_helpers.load_data_and_labels(FLAGS.train_file)
    dev_x_str,dev_target_str, dev_y = data_helpers.load_data_and_labels(FLAGS.test_file)
    test_x_str, test_target_str, test_y = data_helpers.load_data_and_labels(FLAGS.test_file)

    #word embedding ---> x[324,1413,1,41,43,0,0,0]  y[0,1]
    #word_id_mapping,such as  apple--->23 ,w2v  23---->[vector]
    word_id_mapping, w2v = data_helpers.load_w2v(FLAGS.embedding_file_path, FLAGS.word_embedding_dim)
    max_document_length = max([len(x.split(" ")) for x in (train_x_str + dev_x_str + test_x_str)])
    max_target_length = max([len(x.split(" ")) for x in (train_target_str + dev_target_str + test_target_str)])

    #The targets  ---->[[[141,23,45],[23,45,1,2],[2]], ...]
    #The number of targets ----> [3, ...]
    train_targets_str,train_targets_num = data_helpers.load_targets(FLAGS.train_file)
    dev_targets_str,dev_targets_num = data_helpers.load_targets(FLAGS.test_file)
    test_targets_str, test_targets_num = data_helpers.load_targets(FLAGS.test_file)
    max_target_num = max([len(x) for x in (train_targets_str + test_targets_str)])

    # sentence ---> word_id
    train_x, train_x_len = data_helpers.word2id(train_x_str,word_id_mapping,max_document_length)
    dev_x, dev_x_len = data_helpers.word2id(dev_x_str,word_id_mapping,max_document_length)
    test_x, test_x_len = data_helpers.word2id(test_x_str,word_id_mapping,max_document_length)
    # target ---> word_id
    train_target, train_target_len = data_helpers.word2id(train_target_str,word_id_mapping,max_target_length)
    dev_target, dev_target_len = data_helpers.word2id( dev_target_str,word_id_mapping,max_target_length)
    test_target, test_target_len = data_helpers.word2id(test_target_str,word_id_mapping,max_target_length)
    # targets ---> word_id
    train_targets, train_targets_len = data_helpers.word2id_2(train_targets_str,word_id_mapping,max_target_length,max_target_num)
    dev_targets, dev_targets_len = data_helpers.word2id_2(dev_targets_str,word_id_mapping,max_target_length,max_target_num)
    test_targets, test_targets_len = data_helpers.word2id_2(test_targets_str,word_id_mapping,max_target_length,max_target_num)

    #which one targets in all targets
    train_target_whichone = data_helpers.get__whichtarget(train_targets_num, max_target_num)
    test_target_whichone = data_helpers.get__whichtarget(test_targets_num, max_target_num)
    # target position
    train_target_position  = data_helpers.get_position(FLAGS.train_file,max_document_length)
    test_target_position  = data_helpers.get_position(FLAGS.test_file,max_document_length)

    train_targets_position  = data_helpers.get_position_2(train_target_position,train_targets_num,max_target_num)
    test_targets_position  = data_helpers.get_position_2(test_target_position,test_targets_num,max_target_num)

    #Relation Matrix
    #use test_target to creat the relation
    train_relation_self,train_relation_cross = data_helpers.get_relation(train_targets_num, max_target_num,FLAGS.which_relation)
    test_relation_self, test_relation_cross = data_helpers.get_relation(test_targets_num, max_target_num,FLAGS.which_relation)
    Train = {'x':train_x,                       # int32(3608, 79)       train sentences input embeddingID
             'T':train_target,                  # int32(3608, 23)       train target input embeddingID
             'Ts':train_targets,                # int32(3608, 13, 23)   train targets input embeddingID
             'x_len':train_x_len,               # int32(3608,)          train sentences input len
             'T_len':train_target_len,          # int32(3608,)          train target len
             'Ts_len': train_targets_len,       # int32(3608, 13)       train targets len
             'T_W': train_target_whichone,      # int32(3608, 13)       the ith number of all the targets
             'T_P':train_target_position,       # float32(3608, 79)
             'Ts_P': train_targets_position,    # float32(3608,13, 79)
             'R_Self': train_relation_self,     # int32(3608, 13, 13)
             'R_Cross': train_relation_cross,   # int32(3608, 13, 13)
             'y': train_y,  # int32(3608, 3)
            }
    Test = { 'x':test_x,
             'T':test_target,
             'Ts':test_targets,
             'x_len':test_x_len,
             'T_len':test_target_len,
             'Ts_len': test_targets_len,
             'T_W': test_target_whichone,
             'T_P': test_target_position,
             'Ts_P': test_targets_position,
             'R_Self': test_relation_self,
             'R_Cross': test_relation_cross,
             'y': test_y,
            }
    #
    # batches = data_helpers.batch_iter(
    #     list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    print("Vocabulary Size: {:d}".format(len(word_id_mapping)))
    print("Train/Dev/test split: {:d}/{:d}/{:d}".format(len(train_y), len(dev_y), len(test_y)))
    return Train,Test, w2v


def train(Train, Test, word_embedding):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = eval(use_model)(
                sequence_length=Train['x'].shape[1],
                target_sequence_length = Train['T'].shape[1],
                targets_num_max = Train['Ts'].shape[1],
                num_classes=Train['y'].shape[1],
                word_embedding = word_embedding,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            writer = tf.summary.FileWriter("logs/LSTM_GCN3", sess.graph)

            vs = tf.trainable_variables()
            print('There are %d train_able_variables in the Graph: ' % len(vs))
            for v in vs:
                print(v)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", use_data,use_model))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Test summaries
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # else:
            #     raise Exception('The checkpoint_dir already exists:',checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch,T_batch,Ts_batch,x_len_batch,T_len_batch,Ts_len_batch,R_Self_batch,R_Cross_batxh,T_W_batch,T_P_batch,Ts_P_batch,y_batch):
                """
                A single training step
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_target:T_batch,
                    model.input_targets_all:Ts_batch,
                    model.sen_len:x_len_batch,
                    model.target_len:T_len_batch,
                    model.targets_all_len_a:Ts_len_batch,
                    model.relate_self:R_Self_batch,
                    model.relate_cross:R_Cross_batxh,
                    model.target_which:T_W_batch,
                    model.target_position: T_P_batch,
                    model.targets_all_position_a: Ts_P_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def test_step(x_batch,T_batch,Ts_batch,x_len_batch,T_len_batch,Ts_len_batch,R_Self_batch,R_Cross_batxh,T_W_batch,T_P_batch,Ts_P_batch,y_batch, summary = None,writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_target:T_batch,
                    model.input_targets_all:Ts_batch,
                    model.sen_len:x_len_batch,
                    model.target_len:T_len_batch,
                    model.targets_all_len_a:Ts_len_batch,
                    model.relate_self:R_Self_batch,
                    model.relate_cross:R_Cross_batxh,
                    model.target_which: T_W_batch,
                    model.target_position: T_P_batch,
                    model.targets_all_position_a: Ts_P_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, softmax,true_y,predictions = sess.run(
                    [global_step, summary, model.loss, model.accuracy, model.softmax,model.true_y, model.predictions],
                    feed_dict)
                F1 = metrics.f1_score(true_y, predictions, average='macro')
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy,F1))
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy,softmax,F1

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(Train['x'],Train['T'],Train['Ts'],Train['x_len'], Train['T_len'], Train['Ts_len'],
                         Train['R_Self'],Train['R_Cross'],Train['T_W'],Train['T_P'],Train['Ts_P'],Train['y'])), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            train_acc, dev_acc, test_acc, train_all_softmax, test_all_softmax = [], [], [], [], []
            max_test_acc = 0
            max_test_F1_macro = 0
            for batch in batches:
                x_batch,T_batch,Ts_batch,x_len_batch,T_len_batch,Ts_len_batch,R_Self_batch,R_Cross_batxh,T_W_batch,T_P_batch,Ts_P_batch,y_batch = zip(*batch)
                train_step(x_batch,T_batch,Ts_batch,x_len_batch,T_len_batch,Ts_len_batch,R_Self_batch,R_Cross_batxh,T_W_batch,T_P_batch,Ts_P_batch,y_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    print('\nBy now ,the max test acc is: ', max_test_acc)
                    print('        the max F1 score is: ', max_test_F1_macro)
                    print("\nEvaluation Text:")
                    test_acc_i, test_softmax_i, test_F1_i = test_step(Test['x'],Test['T'],Test['Ts'],Test['x_len'], Test['T_len'], Test['Ts_len'],
                                                           Test['R_Self'],Test['R_Cross'],Test['T_W'],Test['T_P'],Test['Ts_P'],Test['y'], summary = test_summary_op, writer=test_summary_writer)
                    test_acc.append(test_acc_i)
                    test_all_softmax.append(test_softmax_i)
                    print('----------------------------------------------------------')
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    if test_acc_i>max_test_acc:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                        print('->>>>>>>>>>>>>>>>>>>>>>>')
                        max_test_step = current_step
                        max_test_acc = test_acc_i
                    if test_F1_i > max_test_F1_macro:
                        max_test_F1_macro = test_F1_i

            print('max_test_step: ', max_test_step)
            print('max_test_acc: ', max_test_acc)
            print('max_test_F1_macro: ', max_test_F1_macro)
    return train_acc, dev_acc, max_test_acc,max_test_F1_macro,max_test_step, train_all_softmax, test_all_softmax


if __name__ == '__main__':
	#文件处理
	Train, Test, word_embedding = preprocess()
	#模型训练
	train_acc, dev_acc, max_test_acc,max_test_F1_macro,max_test_step, train_all_softmax, test_all_softmax = train(Train, Test, word_embedding)

