# -*-coding:utf-8-*-
import argparse
import os
from copy import deepcopy

import numpy as np
import tensorflow as tf

import resolve_data as rd
from code_generate_model import code_gen_model
# resolve_data Functions
from resolve_data import batch_data, get_classnum, readrules, resolve_data, rulebondast, tqdm
# resolve_data Variables
from resolve_data import char_vocabulary, tree_vocabulary, vocabulary

# XXX Instance variables of a model/trainer class might be an improvament
global project, embedding_size, conv_layernum, conv_layersize, rnn_layernum
global cardnum, copynum, copylst
global batch_size, learning_rate, keep_prob, pretrain_times
global rulelist_len, rules_len, NL_len, Tree_len, parent_len


def pre_mask():
    mask = np.zeros([rulelist_len, rulelist_len])
    for i in range(rulelist_len):
        for t in range(i + 1):
            mask[i][t] = 1
    return mask


def get_card(lst):
    global cardnum
    global copynum
    global copylst
    if True:  # len(cardnum) == 0:
        f = open(project + "nlnum.txt", "r")
        st = f.read()
        cardnum = eval(st)
        f.close()
    if True:  # copynum == 0:
        f = open(project + "copylst.txt", "r")
        st = f.read()
        copylst = eval(st)
        for x in copylst:
            if x == 1:
                copynum += 1

    dic = {}
    copydic = {}
    wrongnum = 0
    wrongcnum = 0
    for i, x in enumerate(lst):
        if x == False:
            if copylst[i] == 1:
                wrongnum += 1
                if cardnum[i] not in copydic:
                    wrongcnum += 1
                    copydic[cardnum[i]] = 1
            if cardnum[i] not in dic:
                dic[cardnum[i]] = 1
    devs_num = 0
    if "ATIS" in project:
        devs_num = 491
    elif "HS" in project:
        devs_num = 66
    return devs_num - len(dic), wrongnum / copynum, wrongcnum


def create_model(session, g, placeholder=""):
    if (os.path.exists(project + "save1")):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(project + "save1/"))
        print("load the model")
    else:
        session.run(tf.global_variables_initializer(), feed_dict={})
        print("create a new model")


def save_model(session, number):
    saver = tf.train.Saver()
    saver.save(session, project + "save" + str(number) + "/model.cpkt")


def save_model_time(session, number, card):
    saver = tf.train.Saver()
    saver.save(session, project + "save_list/save" + str(number) + "_" + str(card) + "/model.cpkt")


def get_state(batch_data):
    vec = np.zeros([len(batch_data[6])])
    for i in range(len(batch_data[6])):
        index = 79
        for t in range(len(batch_data[6][i])):
            if batch_data[6][i][t] == 0:
                index = t - 1
                break
        vec[i] = index
    return vec


def g_pretrain(sess, model, batch_data):
    batch = deepcopy(batch_data)
    rewards = np.zeros([len(batch[1])])

    for i in range(len(rewards)):
        rewards[i] = 1

    loss_mask = np.zeros([len(batch[9]), len(batch[9][0])])
    for i in range(len(batch[9])):
        loss_mask[i][0] = 1
        for t in range(1, len(batch[9][i])):
            if batch[9][i][t] == 0:
                break
            loss_mask[i][t] = 1

    state = get_state(batch_data)

    _, pre, a = sess.run([model.optim, model.correct_prediction, model.cross_entropy],
                         feed_dict={model.input_NL: batch[0],
                                    model.input_NLChar: batch[1],
                                    model.inputparentlist: batch[5],
                                    model.inputrulelist: batch[6],
                                    model.inputrulelistnode: batch[7],
                                    model.inputrulelistson: batch[8],
                                    model.inputY_Num: batch[9],
                                    model.tree_path_vec: batch[12],
                                    model.labels: batch[18],
                                    model.loss_mask: loss_mask,
                                    model.antimask: pre_mask(),
                                    model.treemask: batch[16],
                                    model.father_mat: batch[17],
                                    model.state: state,
                                    model.keep_prob: 0.85,
                                    model.rewards: rewards,
                                    model.is_train: True
                                    })
    return pre


def rules_component_batch(batch):
    vecnode = np.zeros([len(batch[2]), len(batch[2][0])])
    vecson = np.zeros([len(batch[2]), len(batch[2][0]), 3])
    # print (batch[2])
    # print (batch[-1])
    for i in range(len(batch[2])):
        for t in range(len(batch[2][0])):
            vnode, vson = rulebondast(int(batch[2][i][t]), "", batch[-1][i])
            vecnode[i, t] = vnode[0]
            for q in range(3):
                vecson[i, t, q] = vson[q]
    return [vecnode, vecson]


def g_eval(sess, model, batch_data):
    batch = batch_data
    rewards = np.zeros([len(batch[1])])

    for i in range(len(rewards)):
        rewards[i] = 1

    loss_mask = np.zeros([len(batch[9]), len(batch[9][0])])
    for i in range(len(batch[9])):
        loss_mask[i][0] = 1
        for t in range(1, len(batch[9][i])):
            if batch[9][i][t] == 0:
                break
            loss_mask[i][t] = 1

    state = get_state(batch_data)
    acc, pre, pre_rules = sess.run([model.accuracy, model.correct_prediction, model.max_res],
                                   feed_dict={model.input_NL: batch[0],
                                              model.input_NLChar: batch[1],
                                              model.inputparentlist: batch[5],
                                              model.inputrulelist: batch[6],
                                              model.inputrulelistnode: batch[7],
                                              model.inputrulelistson: batch[8],
                                              model.inputY_Num: batch[9],
                                              model.tree_path_vec: batch[12],
                                              model.loss_mask: loss_mask,
                                              model.antimask: pre_mask(),
                                              model.treemask: batch[16],
                                              model.father_mat: batch[17],
                                              model.labels: batch[18],
                                              model.state: state,
                                              model.keep_prob: 1,
                                              model.rewards: rewards,
                                              model.is_train: False
                                              })
    p = []
    max_res = []
    for i in range(len(batch[9])):
        for t in range(rules_len):
            if batch[6][i][t] != 0:
                p.append(pre[i][t])
                max_res.append(pre_rules[i][t])
            else:
                p.pop()
                max_res.pop()
                break
    return acc, p, max_res


def run():
    Code_gen_model = code_gen_model(get_classnum(), embedding_size, conv_layernum, conv_layersize, rnn_layernum,
                                    batch_size, len(vocabulary), len(tree_vocabulary), NL_len, Tree_len,
                                    rd.parent_len, learning_rate, keep_prob, len(char_vocabulary),
                                    rules_len)
    valid_batch, _ = batch_data(batch_size, "dev")  # read data
    best_accuracy = 0
    best_card = 0
    config = tf.ConfigProto(allow_soft_placement=True)  # , log_device_placement=True)
    config.gpu_options.allow_growth = True
    f = open(project + "out.txt", "w")
    with tf.Session(config=config) as sess:
        create_model(sess, Code_gen_model, "")
        best_time = -1
        for i in tqdm(range(pretrain_times)):
            Code_gen_model.steps += 1.
            batch, _ = batch_data(batch_size, "train")
            for j in tqdm(range(len(batch))):
                if i % 3 == 0 and j % 2000 == 0:  # eval
                    ac = 0
                    res = []
                    sumac = 0
                    length = 0
                    for k in range(len(valid_batch)):
                        ac1, loss1, _ = g_eval(sess, Code_gen_model, valid_batch[k])

                        res.extend(loss1)
                        ac += ac1;
                    ac /= len(valid_batch)
                    card, copyc, copycard = get_card(res)
                    strs = str(ac) + " " + str(card) + "\n"
                    f.write(strs)
                    f.flush()

                    print("current accuracy " +
                          str(ac) + " string accuarcy is " + str(card))
                    if best_card < card:
                        best_card = card
                        best_accuracy = ac
                        save_model(sess, 1)
                        best_time = i
                        print("find the better accuracy " +
                              str(best_accuracy) + "in epoches " + str(i))
                    elif card == best_card:
                        if (best_accuracy < ac):
                            best_card = card
                            best_accuracy = ac
                            save_model(sess, 1)
                            print("find the better accuracy " +
                                  str(best_accuracy) + "in epoches " + str(i))

                if i % 50 == 0 and j == 0:
                    ac = 0
                    res = []
                    sumac = 0
                    length = 0
                    for k in range(len(valid_batch)):
                        ac1, loss1, _ = g_eval(sess, Code_gen_model, valid_batch[k])

                        res.extend(loss1)
                        ac += ac1;
                    print(len(res))
                    ac /= len(valid_batch)
                    card, copyc, copycard = get_card(res)

                    print("current accuracy " +
                          str(ac) + " string accuracy is " + str(card))
                    save_model_time(sess, i, str(int(Code_gen_model.steps)))
                g_pretrain(sess, Code_gen_model, batch[j])
                tf.train.global_step(sess, Code_gen_model.global_step)

    f.close()
    # print("training finish")
    return


def get_args():
    parser = argparse.ArgumentParser("Train model", fromfile_prefix_chars='@')
    parser.add_argument("project_name", type=str)

    parser.add_argument("--embedding-size", type=int, default=None)
    parser.add_argument("--conv-layernum", type=int, default=256)
    parser.add_argument("--conv-layersize", type=int, default=3)
    parser.add_argument("--rnn-layernum", type=int, default=50)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--keep-prob", type=int, default=0.8)
    parser.add_argument("--pretrain-times", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None)

    return parser.parse_args()


def main():
    args = get_args()

    global project, embedding_size, conv_layernum, conv_layersize, rnn_layernum
    global batch_size, learning_rate, keep_prob, pretrain_times
    global cardnum, copynum, copylst
    global rulelist_len, rules_len, NL_len, Tree_len, parent_len

    project = args.project_name + "/"
    embedding_size = args.embedding_size if args.embedding_size is not None else 256 if "HS" in project else 128
    conv_layernum = args.conv_layernum
    conv_layersize = args.conv_layersize
    rnn_layernum = args.rnn_layernum
    batch_size = args.batch_size if args.batch_size is not None else 64 if "HS" in project else 30
    learning_rate = args.learning_rate
    keep_prob = args.keep_prob
    pretrain_times = args.pretrain_times

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    cardnum = []
    copynum = 0
    copylst = []

    readrules(project)
    resolve_data(project)

    # XXX Why duplicate variable names and values from resolve_data?
    # XXX Instance variables of a class for datasets might be an improvement
    rulelist_len = rd.rulelist_len
    rules_len = rd.rules_len
    NL_len = rd.nl_len
    Tree_len = rd.tree_len
    parent_len = rd.parent_len

    run()


if __name__ == '__main__':
    main()
