#sample 300 sentence from traindata and value the performance
import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
import cnnmodel_union_transformer_with_cnn_entity_desc_twoLoss as model_test
import pickle
import sys

#reload(sys)
#sys.setdefaultencoding("utf-8")

def produce_label_data():

    with open('origin_data/test_type.csv','r',encoding= 'utf-8') as input:
        test_data = input.readlines()

    dict_relation2id = {}#{realtion:id}
    label_entitypair = {}#{e1$e2, set[relation1,relation2,realtion3]}


    with open('origin_data/relation2id.txt','r',encoding='utf-8') as input:
        lines = input.readlines()

    for line in lines:
        line = line.strip()
        relation = line.split()[0]
        id = int (line.split()[1])
        dict_relation2id[relation] = id

    for line in test_data:
        line = line.strip()
        items = line.split('|')
        e1 = items[0]
        e2 = items[1]
        relationid =dict_relation2id[items[6]]
        key = e1+'$'+e2
        if key not in label_entitypair.keys():
            label_entitypair[key] = set()
            label_entitypair[key].add(relationid)
        else:
            label_entitypair[key].add(relationid)

    num_entitypair = len(label_entitypair)
    num_entitypair_true = 0
    for key in label_entitypair.keys():
        tmp_set = label_entitypair[key]
        if len(tmp_set) >1:
            num_entitypair_true+=1
        elif 0 not in tmp_set:
            num_entitypair_true += 1

    print ('num_entitypair:',num_entitypair)
    print ('num_entitypair_true:',num_entitypair_true)

    with open('entity_type/data/label_entitypair.pkl','wb') as output:
        pickle.dump(label_entitypair,output)
    return (num_entitypair_true)


def produce_pred_data(save_path ,output_path):

    test_word = np.load('entity_type/cnndata/cnn_test_word.npy')
    test_pos1 = np.load('entity_type/cnndata/cnn_test_pos1.npy')
    test_pos2 = np.load('entity_type/cnndata/cnn_test_pos2.npy')
    test_y = np.load('entity_type/cnndata/cnn_test_y.npy')
    test_type1 = np.load('entity_type/cnndata/cnn_test_entity1.npy')
    test_type2 = np.load('entity_type/cnndata/cnn_test_entity2.npy')
    with open('origin_data/test_type.csv','r',encoding= 'utf-8') as input:
        test_data = input.readlines()
    with open( 'entity_type/data/label_entitypair.pkl', 'rb') as input:
        label_entitypair = pickle.load(input)
    
    test_word = np.reshape(test_word, [-1, 70])
    test_pos1 = np.reshape(test_pos1, [-1, 70])
    test_pos2 = np.reshape(test_pos2, [-1, 70])


    pred_entitypair = {}  # {e1$e2, (max_prob,relation)}
    batch_size = 100
    steps = len(test_y)//batch_size +1

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            interact = model_test.interaction(sess, save_path)
    index = 0
    for step in tqdm(range(steps)):
        batch_word = test_word[batch_size*step:batch_size*(step+1)]
        batch_pos1 = test_pos1[batch_size*step:batch_size*(step+1)]
        batch_pos2 = test_pos2[batch_size*step:batch_size*(step+1)]
        batch_type1 = test_type1[batch_size*step:batch_size*(step+1)]
        batch_type2 = test_type2[batch_size*step:batch_size*(step+1)]
        batch_y = test_y[batch_size * step:batch_size * (step + 1)]
        batch_y = [np.argmax(i) for i in batch_y]
        batch_test_date = test_data[batch_size*step:batch_size*(step+1)]
        batch_entitypair = []
        #index = 0
        for line in batch_test_date:
            items = line.split('|')
            e1 = items[0]
            e2 = items[1]
            batch_entitypair.append(e1+'$'+e2)
        batch_relation,batch_prob = interact.test(batch_word,batch_pos1,batch_pos2,batch_type1,batch_type2)
        assert(len(batch_relation) == len(batch_prob) and len(batch_relation) == len(batch_entitypair))
        for i in range(len(batch_relation)):
            label_relations = label_entitypair[batch_entitypair[i]]         
            if batch_relation[i] != 0:
                tmp_key = batch_entitypair[i]
                tmp_value = (batch_prob[i],batch_relation[i])
                if tmp_key not in pred_entitypair.keys():
                    pred_entitypair[tmp_key] = []
                    pred_entitypair[tmp_key] = tmp_value
                elif tmp_value[0] > pred_entitypair[tmp_key][0]:
                    pred_entitypair[tmp_key] = tmp_value
    with open(output_path,'wb') as output:
        pickle.dump(pred_entitypair,output)


def P_N(label_path,pred_path):

    with open(label_path, 'rb') as input:
        label_entitypair = pickle.load(input)
    with open(pred_path, 'rb') as input:
        pred_entitypair = pickle.load(input)

    list_pred = []
    for key in pred_entitypair.keys():
        tmp_prob = pred_entitypair[key][0]
        tmp_relation = pred_entitypair[key][1]
        tmp_entitypair = key
        list_pred.append((tmp_prob,tmp_entitypair,tmp_relation))
    list_pred = sorted(list_pred,key=lambda x:x[0],reverse=True)

    list_pred = list_pred[:501]


    true_positive = 0
    result = []
    for i,item in enumerate(list_pred):
        tmp_entitypair = item[1]
        tmp_relation = item[2]
        label_relations = label_entitypair[tmp_entitypair]
        if tmp_relation in label_relations :
            true_positive+=1
        if i %100==0 and i!=0:
            result.append(float (true_positive/i))
    return  result
    #print ('\n')

def PR_curve(label_path,pred_path,num_total):

    with open(label_path, 'rb') as input:
        label_entitypair = pickle.load(input)
    with open(pred_path, 'rb') as input:
        pred_entitypair = pickle.load(input)

    list_pred = []
    for key in pred_entitypair.keys():
        tmp_prob = pred_entitypair[key][0]
        tmp_relation = pred_entitypair[key][1]
        tmp_entitypair = key
        list_pred.append((tmp_prob,tmp_entitypair,tmp_relation))
    list_pred = sorted(list_pred,key=lambda x:x[0],reverse=True)

    list_pred = list_pred[:2001]

    true_positive = 0
    Precision = []
    Recall = []
    for i, item in enumerate(list_pred):
        tmp_entitypair = item[1]
        tmp_relation = item[2]
        label_relations = label_entitypair[tmp_entitypair]
        if tmp_relation in label_relations:
            true_positive += 1
        if i % 10 == 0 and i != 0:
            Precision.append(true_positive / i)
            Recall.append(true_positive/num_total)
    return (Precision,Recall)

if __name__ == '__main__':

    num_total = produce_label_data()

    produce_pred_data(save_path='entity_type/bestmodel/cnnmodel_union_transformer_with_cnn_with_entity_desc_0.1_0.2_.ckpt',output_path = 'entity_type/result/origin_pred_entitypair.pkl')
    result = P_N(label_path = 'entity_type/data/label_entitypair.pkl',pred_path ='entity_type/result/origin_pred_entitypair.pkl')
    print('origin_cnn_P@100,200,300:', result)

