# -*- coding: utf-8 -*-
'''
# @Author  : plzhao
# @Software: PyCharm
'''
from bert_embedding import BertEmbedding
import numpy as np
import time


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


def get_targets_array(target_array,targets_num,max_target_num):
    """
    结合输入的target_position以及target_num,target_num是多少，就由多少个，并且重复多少次。
    不足max_target_num的，补0.
    """
    positions = []
    i = 0
    while i < targets_num.shape[0] :
        i_position = []
        for t_num in range(targets_num[i]):
            i_position.append(target_array[i+t_num])

        for j in range(max_target_num - targets_num[i]):
            i_position.append(np.zeros([target_array.shape[1],target_array.shape[2]]))
        for t_num in range(targets_num[i]):
            positions.append(i_position)
            i += 1

    return np.array(positions)



#-----------------------Restaurants--------------------------
print('-----------------------Restaurants--------------------------')
train_file = "data_res/bert_embedding/Restaurants_Train_bert.txt"
test_file = "data_res/bert_embedding/Restaurants_Test_bert.txt"

train_target_load_file = "data_res/bert_embedding/Res_Train_target_Embedding.npy"
test_target_load_file = "data_res/bert_embedding/Res_Test_target_Embedding.npy"
train_targets_save_file = "data_res/bert_embedding/Res_Train_targets_Embedding.npy"
test_target_save_file = "data_res/bert_embedding/Res_Test_targets_Embedding.npy"

print("loading data:")

train_targets_str, train_targets_num = load_targets(train_file)
test_targets_str, test_targets_num = load_targets(test_file)
max_target_num = max([len(x) for x in (train_targets_str + test_targets_str)])

train_target_array = np.load(train_target_load_file)
test_target_array = np.load(test_target_load_file)      #([1120,23,768])
train_targets_array = get_targets_array(train_target_array,train_targets_num,max_target_num)
test_targets_array = get_targets_array(test_target_array,test_targets_num,max_target_num)       #([1120,13,23,768])

np.save(train_targets_save_file,train_targets_array)
np.save(test_target_save_file,test_targets_array)
print("finish save --targets array-- in: ", train_targets_save_file)
print("finish save --targets array-- in: ", test_target_save_file)
print()



#-----------------------Laptops--------------------------
print('-----------------------Laptops--------------------------')
train_file = "data_lap/bert_embedding/Laptops_Train_bert.txt"
test_file = "data_lap/bert_embedding/Laptops_Test_bert.txt"

train_target_load_file = "data_lap/bert_embedding/Lap_Train_target_Embedding.npy"
test_target_load_file = "data_lap/bert_embedding/Lap_Test_target_Embedding.npy"
train_targets_save_file = "data_lap/bert_embedding/Lap_Train_targets_Embedding.npy"
test_target_save_file = "data_lap/bert_embedding/Lap_Test_targets_Embedding.npy"

print("loading data:")

train_targets_str, train_targets_num = load_targets(train_file)
test_targets_str, test_targets_num = load_targets(test_file)
max_target_num = max([len(x) for x in (train_targets_str + test_targets_str)])

train_target_array = np.load(train_target_load_file)
test_target_array = np.load(test_target_load_file)
train_targets_array = get_targets_array(train_target_array,train_targets_num,max_target_num)
test_targets_array = get_targets_array(test_target_array,test_targets_num,max_target_num)

np.save(train_targets_save_file,train_targets_array)
np.save(test_target_save_file,test_targets_array)
print("finish save --targets array-- in: ", train_targets_save_file)
print("finish save --targets array-- in: ", test_target_save_file)
print()