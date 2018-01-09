# -*-coding:utf-8-*-
#! /usr/bin/env python

from bilstm import BiLSTM
from _imports import *

import random
import tensorflow as tf
import numpy as np
import sys
import os
import time
import codecs


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_integer("max_sent_length", 200, "Maximum sentence length for padding (default:100)")
tf.flags.DEFINE_integer("num_hidden", 200, "Number of hidden units per layer (default:200)")
tf.flags.DEFINE_integer("num_classes", 2, "Number of y classes (default:2)")
# Training parameters
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate(default: 1e-4)")
tf.flags.DEFINE_boolean("scope_detection", True, "True if the task is scope detection or joined scope detection")
tf.flags.DEFINE_integer("POS_emb",2,"0: no POS embeddings; 1: normal POS; 2: universal POS")
tf.flags.DEFINE_boolean("emb_update",False,"True if input embeddings should be updated (default: False)")
tf.flags.DEFINE_boolean("normalize_emb",False,"True to apply L2 regularization on input embeddings (default: False)")
# Data Parameters
tf.flags.DEFINE_string("test_set",'', "Path to the test filename (to use only in test mode")
tf.flags.DEFINE_boolean("pre_training", False, "True to use pretrained embeddings")
tf.flags.DEFINE_string("training_lang",'fr', "Language of the tranining data (default: en)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def store_config(_dir,flags):
    with codecs.open(os.path.join(_dir,'config.ini'),'wb','utf8') as _config:
        for attr, value in sorted(FLAGS.__flags.items()):
            _config.write("{}={}\n".format(attr.upper(), value))

# Timestamp and output dir for current model
fold_name = "%s%s_%d%s" % ('PRE' if FLAGS.pre_training else "noPRE",
'upd' if FLAGS.pre_training and FLAGS.emb_update else '',
FLAGS.POS_emb,str(int(time.time())))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "NegNN","runs", fold_name))
print ("Writing to {}\n".format(out_dir))

# Set checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
store_config(checkpoint_dir,FLAGS)

# Data Preparation
# ==================================================

# Load data
if not FLAGS.pre_training:
    train_set, valid_set, voc, dic_inv = load_train_dev(FLAGS.scope_detection, FLAGS.training_lang, checkpoint_dir)
    vocsize = len(voc['w2idxs'])
    tag_voc_size = len(voc['t2idxs']) if FLAGS.POS_emb == 1 else len(voc['tuni2idxs'])
else:
    train_set, valid_set, dic_inv, pre_emb_w, pre_emb_t = load_train_dev(FLAGS.scope_detection, FLAGS.training_lang, FLAGS.embedding_dim, FLAGS.POS_emb)
    vocsize = pre_emb_w.shape[0]
    tag_voc_size = pre_emb_t.shape[0]

train_lex, train_tags, train_tags_uni, train_cue, _, train_y = train_set
valid_lex, valid_tags, valid_tags_uni, valid_cue, _, valid_y = valid_set

# Training
# ==================================================
def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def padding(l,max_len,pad_idx,x=True):
    if x: 
        pad = [pad_idx]*(max_len-len(l))
        return np.concatenate((l,pad),axis=0)
    else:
        pad = np.array([[1]]*(max_len-len(l)))
        pad = np.concatenate((l,pad),axis=0)
        return np.reshape(pad, (1,200))

def feeder(_bilstm, lex, cue, tags, _y, train = True):
    X = padding(lex, FLAGS.max_sent_length, vocsize - 1)
    C = padding(cue, FLAGS.max_sent_length, 2)
    T = padding(tags, FLAGS.max_sent_length, tag_voc_size - 1)
    Y = padding(np.asarray(map(lambda x: [0] if x == 0 else [1],_y)).astype('int32'),FLAGS.max_sent_length,0,False)
    feed_dict={
        _bilstm.x: X,
        _bilstm.c: C,
        _bilstm.t: T,
        _bilstm.y: Y,
        _bilstm.istate_fw: np.zeros((1, 2*FLAGS.num_hidden)), 
        _bilstm.istate_bw: np.zeros((1, 2*FLAGS.num_hidden)), 
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


clr = FLAGS.learning_rate

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        bi_lstm = BiLSTM(num_hidden=FLAGS.num_hidden,
                num_classes=FLAGS.num_classes,
                voc_dim=vocsize,
                emb_dim=FLAGS.embedding_dim,
                sent_max_len = FLAGS.max_sent_length,
                tag_voc_dim = tag_voc_size,
                tags = True if FLAGS.POS_emb in [1,2] else False,
                external = FLAGS.pre_training,
                update = FLAGS.emb_update)

        # Define Training procedure
        optimizer = tf.train.AdamOptimizer(clr).minimize(bi_lstm.loss)
        #~ optimizer = tf.train.GradientDescentOptimizer(clr).minimize(bi_lstm.loss)

        saver = tf.train.Saver()

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
                
        train_tot_loss = []
        dev_tot_acc = []
        best_f1 = 0.0
        for e in xrange(FLAGS.num_epochs):

            # shuffle
            if FLAGS.POS_emb in [1,2]: shuffle([train_lex, train_tags, train_tags_uni, train_cue, train_y], 20)
            else: shuffle([train_lex,train_cue,train_y], 20)

            # TRAINING STEP
            train_step_loss = []
            dev_tot_acc = []
            tic = time.time()
            for i in xrange(len(train_lex)):
                if FLAGS.POS_emb in [1,2]:
                    train_loss = feeder(bi_lstm, train_lex[i],train_cue[i], train_tags[i] if FLAGS.POS_emb == 1 else train_tags_uni[i], train_y[i])
                else:
                    train_loss = feeder(bi_lstm, train_lex[i], train_cue[i], [], train_y[i])
                # Calculating batch loss
                train_step_loss.append(train_loss)
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./len(train_lex)),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()
            train_tot_loss.append(sum(train_step_loss)/len(train_step_loss))
            print "TRAINING MEAN LOSS: ", train_tot_loss[e]

            # DEVELOPMENT STEP
            pred_dev = []
            gold_dev = []
            for i in xrange(len(valid_lex)):
                viterbi_sequences = feeder(bi_lstm, valid_lex[i],valid_cue[i],valid_tags[i] if FLAGS.POS_emb == 1 else valid_tags_uni[i],valid_y[i],train=False)

                for p in viterbi_sequences[:len(valid_lex[i])]:
                    p = np.reshape(p, (len(valid_lex[i]),1))

                pred_dev.append(p)
                gold_dev.append(np.asarray(map(lambda x: [0] if x == 0 else [1],valid_y[i])))
                
            f1,rep_dev,cm_dev,f1_pos = get_eval(pred_dev,gold_dev)
            dev_tot_acc.append(f1_pos)

            # STORE TRAINING LOSS AND DEV ACCURACIES
            np.save(os.path.join(checkpoint_dir,'train_loss.npy'),train_tot_loss)
            np.save(os.path.join(checkpoint_dir,'valid_acc.npy'),dev_tot_acc)

            #~ # STORE INTERMEDIATE RESULTS
            if f1 > best_f1:
                best_f1 = f1
                print ("Best f1 is: ",best_f1)
                be = e
                # store the model
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                saver.save(sess, checkpoint_prefix,global_step=be)
                print ("Model saved.")
                write_report(checkpoint_dir, rep_dev, cm_dev, 'dev')
                store_prediction(checkpoint_dir, valid_lex, dic_inv, pred_dev, gold_dev, 'dev')
                dry = 0
            else:
                dry += 1

            if abs(be-e) >= 10 and dry>=5:
                print ("Halving the lr...")
                clr *= 0.5
                dry = 0
