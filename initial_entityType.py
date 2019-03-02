import numpy as np
import os
import pickle


# embedding the position
def pos_embed(x):
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag


# reading data
def init_batchdata():

    #train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    #train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector
    #test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    #test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)

    print('reading word embedding data...')
    vec = []
    word2id = {}
    #import the word vec
    f = open('./origin_data/vec_add.txt', encoding='utf-8')
    # info = f.readline()
    # print ('word vec info:',info)
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split(" ")
        word = "".join(content[0:-50])
        word2id[word] = len(word2id)
        # print(word)

        content = content[-50:]
        # print(content)
        # content = content.split('|')
        content = [float(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    dim = 50
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)

    print('reading entity to id ')

    with open('./data/dict_entityname2id.pkl','rb') as input:
        dict_entityname2id = pickle.load(input)


    print('reading relation to id')
    relation2id = {}
    f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()

    entity_category_2id = {}
    f = open('./origin_data/type2id.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        entity_category_2id[content[0]] = int(content[1])
    f.close()




    # length of sentence is 70
    fixlen = 70
    # max length of position embedding is 60 (-60~+60)
    maxlen = 60

    train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector
    train_entity1 = {}
    train_entity2 = {}

    print('reading train data...')
    f = open('./origin_data/train_type.csv', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split("|")
        # get entity name
        en1 = content[2]
        en2 = content[3]

        en1_type = content[4]
        en2_type = content[5]


        en1_type_id = 0
        if content[4] not in entity_category_2id:
            en1_type_id = entity_category_2id['O']
        else:
            en1_type_id = entity_category_2id[content[4]]

        en2_type_id = 0
        if content[5] not in entity_category_2id:
            en2_type_id = entity_category_2id['O']
        else:
            en2_type_id = entity_category_2id[content[5]]


        relation = 0
        if content[6] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[6]]

        # put the same entity pair sentences into a dict
        tup = (en1, en2)
        label_tag = 0
        if tup not in train_sen:
            train_sen[tup] = []
            train_sen[tup].append([])
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            train_ans[tup] = []
            train_ans[tup].append(label)

            en1_id = en1_type_id
            en2_id = en2_type_id
            en1_label_tag = 0
            en2_label_tag = 0
            en1_label = [0 for i in range(len(entity_category_2id))]
            en2_label = [0 for i in range(len(entity_category_2id))]
            en1_label[en1_id] = 1
            en2_label[en2_id] = 1
            train_entity1[tup] = []
            train_entity2[tup] = []
            train_entity1[tup].append(en1_label)
            train_entity2[tup].append(en2_label)
        else:
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1

            en1_id = en1_type_id
            en2_id = en2_type_id
            en1_label_tag = 0
            en2_label_tag = 0
            en1_label = [0 for i in range(len(entity_category_2id))]
            en2_label = [0 for i in range(len(entity_category_2id))]
            en1_label[en1_id] = 1
            en2_label[en2_id] = 1

            temp = find_index(label, train_ans[tup])
            if temp == -1:
                train_ans[tup].append(label)
                train_entity1[tup].append(en1_label)
                train_entity2[tup].append(en2_label)
                label_tag = len(train_ans[tup]) - 1
                en1_label_tag = len(train_entity1[tup]) - 1
                en2_label_tag = len(train_entity2[tup]) - 1
                train_sen[tup].append([])
            else:
                label_tag = temp
                en1_label_tag = temp
                en2_label_tag = temp

        sentence = content[7:]
        sentence = " ".join(sentence)
        sentence = sentence.split(" ")
        en1pos = 0
        en2pos = 0

        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            # print(sentence[i])
            if sentence[i] not in word2id:
                '''
                ps = sentence[i].split('_')
                avg_vec = np.zeros(dim)
                c = 0
                for p in ps:
                    if p in word2id:
                        c += 1
                        avg_vec += vec[word2id[p]]
                if c > 0:
                    avg_vec = avg_vec / c
                    word2id[sentence[i]] = len(word2id)
                    vec.append(avg_vec)
                else:'''
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word

        train_sen[tup][label_tag].append(output)

    print('reading test data ...')

    test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)
    test_entity1 = {}
    test_entity2 = {}
    f = open('./origin_data/test_type.csv', 'r', encoding='utf-8')
    count = 0
    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split("|")
        en1 = content[2]
        en2 = content[3]

        en1_type_id = 0
        if content[4] not in entity_category_2id:
            en1_type_id = entity_category_2id['O']
        else:
            en1_type_id = entity_category_2id[content[4]]

        en2_type_id = 0
        if content[5] not in entity_category_2id:
            en2_type_id = entity_category_2id['O']
        else:
            en2_type_id = entity_category_2id[content[5]]

        relation = 0
        if content[6] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[6]]
        tup = (en1, en2, count)
        count += 1

        if tup not in test_sen:
            test_sen[tup] = []
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            test_ans[tup] = label

            type1_id = en1_type_id
            type1_tag = 0
            type1 = [0 for i in range(len(entity_category_2id))]
            type1[type1_id] = 1
            test_entity1[tup] = type1

            type2_id = en2_type_id
            type2_tag = 0
            type2 = [0 for i in range(len(entity_category_2id))]
            type2[type2_id] = 1
            test_entity2[tup] = type2

        else:
            y_id = relation
            test_ans[tup][y_id] = 1

            type1_id = en1_type_id
            test_entity1[tup][type1_id] = 1

            type2_id = en2_type_id
            test_entity2[tup][type2_id] = 1

        sentence = content[7:]
        sentence = " ".join(sentence)
        sentence = sentence.split(" ")
        en1pos = 0
        en2pos = 0

        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word, rel_e1, rel_e2])

        for i in range(min(fixlen, len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                '''
                ps = sentence[i].split('_')
                avg_vec = np.zeros(dim)
                c = 0
                for p in ps:
                    if p in word2id:
                        c += 1
                        avg_vec += vec[word2id[p]]
                if c > 0:
                    avg_vec = avg_vec / c
                    word2id[sentence[i]] = len(word2id)
                    vec.append(avg_vec)
                else:'''
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word
        test_sen[tup].append(output)



    train_x = []
    train_y = []
    train_entity1_list = []
    train_entity2_list = []
    test_x = []
    test_y = []
    train_entitypair = []
    test_entity1_list = []
    test_entity2_list = []

    print('organizing train data')
    #f = open('./data/train_q&a.txt', 'w', encoding='utf-8')
    temp = 0
    for i in train_sen:
        #print (i)
        #return 0
        en1id =  dict_entityname2id[i[0]]
        en2id =  dict_entityname2id[i[1]]
        tmp_pair = (en1id,en2id)
        if len(train_ans[i]) != len(train_sen[i]):
            print('ERROR')
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])
            train_entity1_list.append(train_entity1[i][j])
            train_entity2_list.append(train_entity2[i][j])

            train_entitypair.append(tmp_pair)

            #f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    #f.close()

    print('organizing test data')
    #f = open('./data/test_q&a.txt', 'w', encoding='utf-8')
    temp = 0
    for i in test_sen:
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])
        test_entity1_list.append(test_entity1[i])
        test_entity2_list.append(test_entity2[i])

        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        #f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + tempstr + '\n')
        temp += 1
    #f.close()

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_entity1_list = np.array(train_entity1_list)
    train_entity2_list = np.array(train_entity2_list)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_entity1_list = np.array(test_entity1_list)
    test_entity2_list = np.array(test_entity2_list)
    train_entitypair = np.array(train_entitypair)

    np.save('./entity_type/data/vec.npy', vec)
    np.save('./entity_type/data/train_x.npy', train_x)
    np.save('./entity_type/data/train_y.npy', train_y)
    np.save('./entity_type/data/train_entity1_list.npy', train_entity1_list)
    np.save('./entity_type/data/train_entity2_list.npy', train_entity2_list)
    np.save('./entity_type/data/testall_x.npy', test_x)
    np.save('./entity_type/data/testall_y.npy', test_y)
    np.save('./entity_type/data/test_entity1_list.npy', test_entity1_list)
    np.save('./entity_type/data/test_entity2_list.npy', test_entity2_list)
    np.save('./entity_type/data/train_entitypair',train_entitypair)

    #print (len(train_x),len(train_entitypair))



def seperate():
    print('reading training data')
    x_train = np.load('./entity_type/data/train_x.npy')

    train_word = []
    train_pos1 = []
    train_pos2 = []

    print('seprating train data')
    for i in range(len(x_train)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_train[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    np.save('./entity_type/data/train_word.npy', train_word)
    np.save('./entity_type/data/train_pos1.npy', train_pos1)
    np.save('./entity_type/data/train_pos2.npy', train_pos2)

    #print (len(train_word))


    print('seperating test data')
    x_test = np.load('./entity_type/data/testall_x.npy')
    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)

    np.save('./entity_type/data/testall_word.npy', test_word)
    np.save('./entity_type/data/testall_pos1.npy', test_pos1)
    np.save('./entity_type/data/testall_pos2.npy', test_pos2)



def init_entityebd():

    dict_entityname2id = {}
    print('reading train data...')
    f = open('./origin_data/train.txt', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '':
            break

        content = content.strip().split('\t')
        # get entity name
        en1 = content[2]
        en2 = content[3]
        if en1 not in dict_entityname2id.keys():
            dict_entityname2id[en1] = len(dict_entityname2id)
        if en2 not in dict_entityname2id.keys():
            dict_entityname2id[en2] = len(dict_entityname2id)

    with open('./data/dict_entityname2id.pkl','wb') as output:
        pickle.dump(dict_entityname2id,output)
    #print (len(set (dict_entityname2id.keys())),len(dict_entityname2id.values()),len(dict_entityname2id))
    #print (set (dict_entityname2id.values()))

def init_cnndata():

    y_train = np.load('./entity_type/data/train_y.npy')

    train_entity1_list = np.load('./entity_type/data/train_entity1_list.npy')
    train_entity2_list = np.load('./entity_type/data/train_entity2_list.npy')

    train_word = np.load('./entity_type/data/train_word.npy')
    train_pos1 = np.load('./entity_type/data/train_pos1.npy')
    train_pos2 = np.load('./entity_type/data/train_pos2.npy')

    y_test = np.load('./entity_type/data/testall_y.npy')
    test_entity1_list = np.load("./entity_type/data/test_entity1_list.npy")
    test_entity2_list = np.load("./entity_type/data/test_entity2_list.npy")

    test_word = np.load('./entity_type/data/testall_word.npy')
    test_pos1 = np.load('./entity_type/data/testall_pos1.npy')
    test_pos2 = np.load('./entity_type/data/testall_pos2.npy')

    cnn_train_word = []
    cnn_train_pos1 = []
    cnn_train_pos2 = []
    cnn_train_y = []
    cnn_train_entity1 = []
    cnn_train_entity2 = []

    cnn_test_word = []
    cnn_test_pos1 = []
    cnn_test_pos2 = []
    cnn_test_y = []
    cnn_test_entity1 = []
    cnn_test_entity2 = []

    for i in range(len(train_word)):
        for j in train_word[i]:
            cnn_train_word.append(j)
            cnn_train_y.append(y_train[i])
            cnn_train_entity1.append(train_entity1_list[i])
            cnn_train_entity2.append(train_entity2_list[i])
            # break

    for i in range(len(train_pos1)):
        for j in train_pos1[i]:
            cnn_train_pos1.append(j)
            # break

    for i in range(len(train_pos2)):
        for j in train_pos2[i]:
            cnn_train_pos2.append(j)
            # break

    for i in range(len(test_word)):
        for j in test_word[i]:
            cnn_test_word.append(j)
            cnn_test_y.append(y_test[i])
            cnn_test_entity1.append(test_entity1_list[i])
            cnn_test_entity2.append(test_entity2_list[i])
            # break

    for i in range(len(test_pos1)):
        for j in test_pos1[i]:
            cnn_test_pos1.append(j)
            # break

    for i in range(len(test_pos2)):
        for j in test_pos2[i]:
            cnn_test_pos2.append(j)
            # break

    cnn_train_word = np.array(cnn_train_word)
    cnn_train_pos1 = np.array(cnn_train_pos1)
    cnn_train_pos2 = np.array(cnn_train_pos2)
    cnn_train_y = np.array(cnn_train_y)

    cnn_test_word = np.array(cnn_test_word)
    cnn_test_pos1 = np.array(cnn_test_pos1)
    cnn_test_pos2 = np.array(cnn_test_pos2)
    cnn_test_y = np.array(cnn_test_y)
    cnn_test_entity1 = np.array(cnn_test_entity1)
    cnn_test_entity2 = np.array(cnn_test_entity2)

    np.save('./entity_type/cnndata/cnn_train_word.npy', cnn_train_word)
    np.save('./entity_type/cnndata/cnn_train_entity1.npy', cnn_train_entity1)
    np.save('./entity_type/cnndata/cnn_train_entity2.npy', cnn_train_entity2)

    np.save('./entity_type/cnndata/cnn_train_pos1.npy', cnn_train_pos1)
    np.save('./entity_type/cnndata/cnn_train_pos2.npy', cnn_train_pos2)
    np.save('./entity_type/cnndata/cnn_train_y.npy', cnn_train_y)

    np.save('./entity_type/cnndata/cnn_test_word.npy', cnn_test_word)
    np.save('./entity_type/cnndata/cnn_test_pos1.npy', cnn_test_pos1)
    np.save('./entity_type/cnndata/cnn_test_pos2.npy', cnn_test_pos2)
    np.save('./entity_type/cnndata/cnn_test_y.npy', cnn_test_y)
    np.save('./entity_type/cnndata/cnn_test_entity1.npy', cnn_test_entity1)
    np.save('./entity_type/cnndata/cnn_test_entity2.npy', cnn_test_entity2)

if __name__  == '__main__':
    init_entityebd()
    init_batchdata()
    seperate()
    init_cnndata()
    print ('finished init!')