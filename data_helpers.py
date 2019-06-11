# -*- coding: utf-8 -*-
import numpy as np
import re
import itertools
from collections import Counter





def load_data_and_labels(positive_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    examples = list(open(positive_data_file, "r").readlines())
    examples = [s.strip() for s in examples]
    # find the input examples
    input = []
    target = []
    for index,i in enumerate(examples):
        if index%3 == 0:
            i_target =examples[index + 1].strip()
            i = i.replace("$T$", i_target)
            input.append(i)
            target.append(i_target)
    x_text = input
    # Generate labels
    lable=[]
    for index,i in enumerate(examples):
        if index%3 == 2:
            if i[0:1]=='1':
                lable.append([1,0,0])
            if i[0:1]=='0':
                lable.append([0,1,0])
            if i[0:1]=='-':
                lable.append([0,0,1])
    y = np.array(lable)
    return [x_text,target, y]


def load_targets(positive_data_file):
    """
    find the same sentence,output all the targets of each sentence.
    output the targets' number of each sentences
    """
    # Load data from files
    examples = list(open(positive_data_file, "r").readlines())
    examples = [s.strip() for s in examples]

    input = []
    target = []
    for index,i in enumerate(examples):
        if index%3 == 0:
            i_target =examples[index + 1].strip()
            i = i.replace("$T$", i_target)
            input.append(i)
            target.append(i_target)
    x_text = input
    # find the same targets
    all_sentence = [s for s in x_text]
    targets_nums = [all_sentence.count(s) for s in all_sentence]
    targets = []
    i = 0
    while i < len(all_sentence):
        num = targets_nums[i]
        target = []
        for j in range(num):
            target.append(examples[(i+j)*3+1])
        for j in range(num):
            targets.append(target)
        i = i+num
    targets_nums = np.array(targets_nums)
    return [targets,targets_nums]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    #np.random.seed(1)
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def batch_iter2(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    #np.random.seed(1)
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1: #3411,3798,4207
            print ('a bad word embedding: {}'.format(line[0]))
            cnt -= 1
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    print (np.shape(w2v))
    word_dict['UNK'] = cnt + 1
    print(word_dict['UNK'], len(w2v))
    return word_dict, w2v


def word2id(input_file, word_id_file, sentence_len, encoding='utf8'):
    word_to_id = word_id_file
    print ('load word-to-id done!')
    sen_id,  sen_len = [], []
    for i in input_file:
        words = i.split(" ")
        sen = len(words)
        sen_len.append(sen)
        words_id = []
        for word in words:
            try:
                words_id.append(word_to_id[word])
            except:
                words_id.append(word_to_id['UNK'])
        sen_id.append(words_id + [0] * (sentence_len - len(words)))

    return np.asarray(sen_id), np.asarray(sen_len)

def word2id_2(input_file, word_id_file, sentence_len,target_len, encoding='utf8'):
    sen_ids = []
    sen_lens = []
    for x in input_file:
        word_to_id = word_id_file
        sen_id,  sen_len = [], []
        for i in x:
            words = i.split(" ")
            sen = len(words)
            sen_len.append(sen)
            words_id = []
            for word in words:
                try:
                    words_id.append(word_to_id[word])
                except:
                    words_id.append(word_to_id['UNK'])
            sen_id.append(words_id + [0] * (sentence_len - len(words)))
        for j in range(target_len - len(x)):
            sen_id.append([0] * sentence_len)
            sen_len.append(0)
        sen_ids.append(sen_id)
        sen_lens.append(sen_len)

    print('load targets-to-id done!')
    return np.asarray(sen_ids), np.asarray(sen_lens)

def get_relation(targets_num,max_target_num,relation_mode = 'adjacent'):
    '''
    :param target_num: a one dimension array:[1,2,1,1,...]
    :param max_target_num: max_target_num is 13 in Res data
    :param relation_mode: 'adjacent','global','rule'
    :return: relation_self_matrix,relation_cross_matrix ,shape = [?,max_target_num,max_target_num]
    '''
    if relation_mode == 'global':
        relation_self_M = np.eye(max_target_num)
        relation_cross_M = np.ones([max_target_num,max_target_num])
        #cross的里面自己和自己的连接
        relation_cross_M = relation_cross_M - relation_self_M
        relation_self = []
        relation_cross = []
        for i in range(targets_num.shape[0]): #i---indicate the i-th example
            # 把一个矩阵的前[N,N]覆盖到大小为[M,M]的全0矩阵上(其实目的就是为了补0)
            #N指的是该矩阵的targets数量，M是最大的targets数量。
            target_i_num = targets_num[i]  #the number of targets in a sentence
            zero_matrix = np.zeros((max_target_num,max_target_num))
            zero_matrix[0:target_i_num,0:target_i_num] = relation_self_M[0:target_i_num,0:target_i_num]
            relation_self_i = zero_matrix
            zero_matrix = np.zeros((max_target_num,max_target_num))
            zero_matrix[0:target_i_num,0:target_i_num] = relation_cross_M[0:target_i_num,0:target_i_num]
            relation_cross_i = zero_matrix
            relation_self.append(relation_self_i)
            relation_cross.append(relation_cross_i)
        relation_self = np.asarray(relation_self)
        relation_cross = np.asarray(relation_cross)

    if relation_mode == 'adjacent':
        relation_self_M = np.eye(max_target_num)
        zero_matrix = np.zeros((max_target_num, max_target_num))
        for j in range(max_target_num):  # j --- indicate the j-th dimension of a matrix
            if j == 0:
                zero_matrix[j,j] = 1
            else:
                zero_matrix[j, j] = 1
                zero_matrix[j-1, j] = 1
                zero_matrix[j, j-1] = 1
        relation_cross_M = zero_matrix
        relation_cross_M = relation_cross_M - relation_self_M
        relation_self = []
        relation_cross = []
        for i in range(targets_num.shape[0]): #i---indicate the i-th example
            # 把一个矩阵的前[N,N]覆盖到大小为[M,M]的全0矩阵上(其实目的就是为了补0)
            #N指的是该矩阵的targets数量，M是最大的targets数量。
            target_i_num = targets_num[i]  #the number of targets in a sentence
            zero_matrix = np.zeros((max_target_num,max_target_num))
            zero_matrix[0:target_i_num,0:target_i_num] = relation_self_M[0:target_i_num,0:target_i_num]
            relation_self_i = zero_matrix
            zero_matrix = np.zeros((max_target_num,max_target_num))
            zero_matrix[0:target_i_num,0:target_i_num] = relation_cross_M[0:target_i_num,0:target_i_num]
            relation_cross_i = zero_matrix
            relation_self.append(relation_self_i)
            relation_cross.append(relation_cross_i)
        relation_self = np.asarray(relation_self)
        relation_cross = np.asarray(relation_cross)
    # if relation_mode == 'rule':
    #     pass
    #this is a future work
    return relation_self,relation_cross

def get__whichtarget(targets_num,max_target_num,):
    '''
    :param target_num: a one dimension array:[1,2,2,1,...]
    :param max_target_num: max_target_num is 13 in Res data
    :return: which_one  ,shape = [?,max_target_num]:[[1,0,0,0,...],
                                                     [1,0,0,0,...],
                                                     [0,1,0,0,...],
                                                     [1,0,0,0,...],
                                                     ...]
    '''
    which_one = np.zeros((targets_num.shape[0], max_target_num))
    #补上位置信息，如果是3，那就补上[1,0,0][0,1,0][0,0,1]
    #做法：根据每个的数字，循环得到对于位置,当然序号加上该值
    i = 0
    while i <targets_num.shape[0]:
        for j in range(targets_num[i]):
            which_one[i,j] = 1
            i += 1
    return which_one


def get_position(input_file,max_len):
    """

    """
    # Load data from files
    examples = list(open(input_file, "r").readlines())
    examples = [s.strip() for s in examples]
    position = []
    for index,i in enumerate(examples):
        if index%3 == 0:
            #找到$T$的位置
            i_input = examples[index].strip().split(' ')
            for index_j,j in enumerate(i_input):
                if "$T$" in j:
                    i_input[index_j] = '$T$'
            i_target =examples[index + 1].strip().split(' ')
            len_input = len(i_input)
            len_target = len(i_target)
            target_position  = i_input.index("$T$")
            #target 前、中、后个数
            target_b_len =  target_position
            target_m_len = len_target
            target_e_len = len_input - target_position - 1
            target_b_list = list(range(1,target_b_len+1))
            target_b_list.reverse()
            target_m_list = [0 for j in range(target_m_len)]
            target_e_list = list(range(1,target_e_len+1))

            #让距离太远的变正0
            Ls = len(target_b_list+target_m_list+target_e_list)
            for index_j,j in enumerate(target_b_list):
                if j>10:
                    target_b_list[index_j] = Ls
            for index_j,j in enumerate(target_e_list):
                if j>10:
                    target_e_list[index_j] = Ls

            i_position = target_b_list+target_m_list+target_e_list
            i_position_encoder = [(1 -j/Ls)  for j in i_position]
            i_position_encoder = i_position_encoder + [0] * (max_len - len(i_position))
            position.append(i_position_encoder)
    position = np.array(position)
    return position



def get_position_2(target_position,targets_num,max_target_num):
    """
    结合输入的target_position以及target_num,target_num是多少，就由多少个，并且重复多少次。
    不足max_target_num的，补0.
    """
    positions = []
    i = 0
    while i < targets_num.shape[0] :
        i_position = []
        for t_num in range(targets_num[i]):
            i_position.append(target_position[i+t_num])

        for j in range(max_target_num - targets_num[i]):
            i_position.append(np.zeros([target_position.shape[1]]))
        for t_num in range(targets_num[i]):
            positions.append(i_position)
            i += 1

    return np.array(positions)