# -*- coding: utf-8 -*-
'''
# @project : GCN
# @Time    : 2019/4/25 21:24
# @Author  : plzhao
# @FileName: layers.py
'''
import numpy as np
import tensorflow as tf


def cnn_layer(inputs, filter_size, strides, padding, random_base, l2_reg, active_func=None, scope_name="conv"):
    w = tf.get_variable(
        name='conv' + scope_name,
        shape=filter_size,
        # initializer=tf.random_normal_initializer(mean=0., stddev=1.0),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='softmax_b' + scope_name,
        shape=[filter_size[-1]],
        # initializer=tf.random_normal_initializer(mean=0., stddev=1.0),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    x = tf.nn.conv2d(inputs, w, strides, padding) + b
    if active_func is None:
        active_func = tf.nn.relu
    return active_func(x)


def dynamic_rnn(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    outputs, state = tf.nn.dynamic_rnn(
        cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )  # outputs -> batch_size * max_len * n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # batch_size * n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)
    return outputs


def bi_dynamic_rnn(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last',dropout = True, dropout_prob=0.5):
    if dropout:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell(num_units=n_hidden, state_is_tuple=True),output_keep_prob=dropout_prob)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell(num_units=n_hidden, state_is_tuple=True),output_keep_prob=dropout_prob)
    else:
        cell_fw=cell(n_hidden)
        cell_bw=cell(n_hidden)
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    if out_type == 'last':
        outputs_fw, outputs_bw = outputs
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
        outputs = tf.concat([outputs_fw, outputs_bw], 2)
    else:
        pass
        outputs = tf.concat(outputs, 2)  # batch_size * max_len * 2n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, 2 * n_hidden]), index)  # batch_size * 2n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)  # batch_size * 2n_hidden
    return outputs


def bi_dynamic_rnn_diff(cell, inputs_fw, inputs_bw, n_hidden, l_fw, l_bw, max_len, scope_name):
    with tf.name_scope('forward_lstm'):
        outputs_fw, state_fw = tf.nn.dynamic_rnn(
            cell(n_hidden),
            inputs=inputs_fw,
            sequence_length=l_fw,
            dtype=tf.float32,
            scope=scope_name
        )
        batch_size = tf.shape(outputs_fw)[0]
        index = tf.range(0, batch_size) * max_len + (l_fw - 1)
        output_fw = tf.gather(tf.reshape(outputs_fw, [-1, n_hidden]), index)  # batch_size * n_hidden

    with tf.name_scope('backward_lstm'):
        outputs_bw, state_bw = tf.nn.dynamic_rnn(
            cell(n_hidden),
            inputs=inputs_bw,
            sequence_length=l_bw,
            dtype=tf.float32,
            scope=scope_name
        )
        batch_size = tf.shape(outputs_bw)[0]
        index = tf.range(0, batch_size) * max_len + (l_bw - 1)
        output_bw = tf.gather(tf.reshape(outputs_bw, [-1, n_hidden]), index)  # batch_size * n_hidden

    outputs = tf.concat([output_fw, output_bw], 1)  # batch_size * 2n_hidden
    return outputs


def stack_bi_dynamic_rnn(cells_fw, cells_bw, inputs, n_hidden, n_layer, length, max_len, scope_name, out_type='last'):
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw(n_hidden) * n_layer, cells_bw(n_hidden) * n_layer, inputs,
        sequence_length=length, dtype=tf.float32, scope=scope_name)
    if out_type == 'last':
        outputs_fw, outputs_bw = tf.split(2, 2, outputs)
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
        outputs = tf.concat([outputs_fw, outputs_bw], 2)
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, 2 * n_hidden]), index)  # batch_size * 2n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)  # batch_size * 2n_hidden
    return outputs


def reduce_mean_with_len(inputs, length):
    """
    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
    return inputs


def softmax_layer(inputs, n_hidden, random_base, keep_prob, l2_reg, n_class, scope_name='1'):
    w = tf.get_variable(
        name='softmax_w' + scope_name,
        shape=[n_hidden, n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_class)), np.sqrt(6.0 / (n_hidden + n_class))),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    b = tf.get_variable(
        name='softmax_b' + scope_name,
        shape=[n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_class))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    with tf.name_scope('softmax'):
        outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        predict = tf.matmul(outputs, w) + b
        predict = tf.nn.softmax(predict)
    return predict


def WXA_Relu(X, A, W, b):
    '''
    :param W: (600,600)
    :param X:  (?,600,targets_num)
    :param A: (targets_num,targets_num)
    :param b: useless
    :return:
    '''
    X_shape_1 = tf.shape(X)[1]
    X_shape_2 = tf.shape(X)[2]
    X_shape_1_ = tf.shape(W)[1]

    # X_trans = tf.transpose(X, [0, 2, 1]) #(?,targets_num,600)
    # X_trans_reshape = tf.reshape(X_trans, [-1, X_shape_1])#(?*targets_num,600)
    # W_X_trans_reshape = tf.matmul(X_trans_reshape, W)       #(?*targets_num,600)
    # W_X_trans = tf.reshape(W_X_trans_reshape, [-1, X_shape_2, X_shape_1_]) #(?,targets_num,600)
    # W_X = tf.transpose(W_X_trans, [0, 2, 1]) #(?,600,targets_num)
    # W_X_A_relu = tf.nn.relu(tf.matmul(W_X, A))

    X_trans = tf.transpose(X, [0, 2, 1]) #(?,targets_num,600)
    W_X_trans = tf.einsum('ijk,kl->ijl', X_trans, W)
    W_X = tf.transpose(W_X_trans, [0, 2, 1]) #(?,600,targets_num)
    W_X_A_relu = tf.nn.relu(tf.matmul(W_X, A))
    return W_X_A_relu

def WXbA_Relu(X, A, W, b):
    '''
    :param W: (600,600)
    :param X:  (?,600,targets_num)
    :param A: (targets_num,targets_num)
    :param b:
    :return:
    '''
    X_shape_1 = tf.shape(X)[1]
    X_shape_2 = tf.shape(X)[2]
    X_shape_1_ = tf.shape(W)[1]
    X_trans = tf.transpose(X, [0, 2, 1]) #(?,targets_num,600)
    W_X_trans = tf.einsum('ijk,kl->ijl', X_trans, W)
    W_X_b_trans_reshape = tf.reshape(W_X_trans,[-1, X_shape_1_])+b #(?*targets_num,600)
    W_X_b_trans = tf.reshape(W_X_b_trans_reshape, [-1, X_shape_2, X_shape_1_]) #(?,targets_num,600)
    W_X_b = tf.transpose(W_X_b_trans, [0, 2, 1]) #(?,600,targets_num)
    W_X_b_A_relu = tf.nn.relu(tf.matmul(W_X_b, A))
    return W_X_b_A_relu