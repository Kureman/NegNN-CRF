# -*-coding:utf-8-*-
#! /usr/bin/env python

from bilstm import BiLSTM
from random import shuffle
from _imports import *

import cPickle
import tensorflow as tf
import numpy as np
import codecs
import sys
import os


# Parameters
# ==================================================
# Model Parameters
tf.flags.DEFINE_string("test_set",'', "Path to the test filename (to use only in test mode")
tf.flags.DEFINE_string("checkpoint_dir",'',"Path to the directory where the last checkpoint is stored")
tf.flags.DEFINE_string("test_name",'current',"Name to assign to report and test files")
tf.flags.DEFINE_string("test_lang",'fr', "fr for french")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def retrieve_flags(config_file):
    _config = codecs.open(config_file,'rb','utf8').readlines()
    return dict([(k.lower(),v) for k,v in map(lambda x: x.strip().split('='),_config)])

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
config_file = os.path.join(FLAGS.checkpoint_dir,'config.ini')
model_flags = retrieve_flags(config_file)

def str2bool(x):
    return True if x == "True" else False

scope_dect = str2bool(model_flags['scope_detection'])
event_dect = str2bool(model_flags['event_detection'])
embedding_dim = int(model_flags['embedding_dim'])
POS_emb = int(model_flags['pos_emb'])
pre_training = str2bool(model_flags['pre_training'])
num_classes = int(model_flags['num_classes'])
max_sent_length = int(model_flags['max_sent_length'])
num_hidden = int(model_flags['num_hidden'])
emb_update = str2bool(model_flags['emb_update'])
tr_lang = model_flags['training_lang']
learning_rate = model_flags['learning_rate']
nepochs = model_flags['num_epochs']

test_files = FLAGS.test_set.split(',')
# Data Preparation
# ==================================================
def unpickle_data(folder):
   with open(os.path.join(folder,'train_dev.pkl'),'rb') as f:
        return cPickle.load(f)

if not pre_training:
        assert FLAGS.test_lang == tr_lang
        _, _, voc, dic_inv = unpickle_data(FLAGS.checkpoint_dir)
	test_lex, test_tags, test_tags_uni, test_cue, _, test_y = load_test(test_files, voc, scope_dect, event_dect, FLAGS.test_lang)
else:
	test_set, dic_inv, pre_emb_w, pre_emb_t = ext_processor.load_test(test_files, scope_dect, event_dect, FLAGS.test_lang, embedding_dim, POS_emb)
        test_lex, test_tags, test_tags_uni, test_cue, _, test_y = test_set

if pre_training:
	vocsize = pre_emb_w.shape[0]
        tag_voc_size = pre_emb_t.shape[0]
else:
        vocsize = len(voc['w2idxs'])
        tag_voc_size = len(voc['t2idxs']) if POS_emb == 1 else len(voc['tuni2idxs'])

# Evaluation
# ==================================================
def padding(l,max_len,pad_idx,x=True):
    if x: 
        pad = [pad_idx]*(max_len-len(l))
        return np.concatenate((l,pad),axis=0)
    else:
        pad = np.array([[1]]*(max_len-len(l)))
        pad = np.concatenate((l,pad),axis=0)
        return np.reshape(pad, (1,200))

def feeder(_bilstm, lex, cue, tags, _y, train = True):
    X = padding(lex, max_sent_length, vocsize - 1)
    C = padding(cue, max_sent_length, 2)
    T = padding(tags, max_sent_length, tag_voc_size - 1)
    Y = padding(np.asarray(map(lambda x: [0] if x == 0 else [1],_y)).astype('int32'), max_sent_length,0,False)
    feed_dict={
        _bilstm.x: X,
        _bilstm.c: C,
        _bilstm.t: T,
        _bilstm.y: Y,
        _bilstm.istate_fw: np.zeros((1, 2*num_hidden)),
        _bilstm.istate_bw: np.zeros((1, 2*num_hidden)),
        _bilstm.sequence_lengths: np.asarray([len(lex)])}
    if train:
        feed_dict.update({_bilstm.lr:clr})
        _, train_loss = sess.run([optimizer, bi_lstm.loss], feed_dict = feed_dict)
        return train_loss
    else:
        sequence_lengths = np.asarray([len(lex)])
        
        logits, trans_params = sess.run([_bilstm.logits, _bilstm.transition_params], feed_dict = feed_dict)
        
        for logit, sequence_length in zip(logits, sequence_lengths):
            
            logit = logit[:sequence_length]
            
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
        
        viterbi_seq = np.asarray([viterbi_seq])
        return viterbi_seq


graph = tf.Graph()
with graph.as_default():
    # session_conf = tf.ConfigProto(
    #   allow_soft_placement=allow_soft_placement,
    #   log_device_placement=log_device_placement)
    sess = tf.Session()
    with sess.as_default():
        bi_lstm = BiLSTM(
                num_hidden=num_hidden,
                num_classes=num_classes,
                voc_dim=vocsize,
                emb_dim=embedding_dim,
                sent_max_len = max_sent_length,
                tag_voc_dim = tag_voc_size,
                tags = True if POS_emb in [1,2] else False,
                external = pre_training,
                update = emb_update)
        saver = tf.train.Saver(tf.global_variables())
        # saver.restore(sess, checkpoint_file)
        # print "Model restored!"

        # load model from last checkpoint
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        saver.restore(sess,checkpoint_file)
        print "Model restored!"
        # Collect the predictions here
        test_tot_acc = []
        preds_test, gold_test = [],[]
        for i in xrange(len(test_lex)):
            pred_test = feeder(bi_lstm, test_lex[i], test_cue[i], test_tags[i] if POS_emb == 1 else test_tags_uni[i],test_y[i], train = False)
            
            for p in pred_test[:len(test_lex[i])]:
                p = np.reshape(p, (len(test_lex[i]),1))
                
            preds_test.append(p)
            gold_test.append(np.asarray(map(lambda x: [0] if x == 0 else [1], test_y[i])))
            
        print 'Mean test accuracy: ', sum(test_tot_acc)/len(test_lex)
        f1,report_tst,best_test, f1_pos = get_eval(preds_test,gold_test)
        
        write_report(FLAGS.checkpoint_dir,report_tst,best_test,FLAGS.test_name)
        store_prediction(FLAGS.checkpoint_dir, test_lex, dic_inv, preds_test, gold_test,FLAGS.test_name)
