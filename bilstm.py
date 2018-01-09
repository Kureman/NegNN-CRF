#-*- coding: utf-8-*-
import tensorflow as tf

import numpy as np
import random
import codecs
import os,sys
import time
import subprocess

def random_uniform(shape,name,low=-1.0,high=1.0,update=True):
    return  tf.Variable(0.2 * tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32),name=name,trainable=update)

class BiLSTM(object):
    def __init__(self,
            num_hidden,
            num_classes,
            voc_dim,
            emb_dim,
            sent_max_len,
            tag_voc_dim,
            tags,
            external,
            update):

        # tf Graph
        # shared parameters
        self.num_hidden = num_hidden
        self.sent_max_len = sent_max_len
        self.num_classes = num_classes
        # input placeholders
        self.sequence_lengths = tf.placeholder(tf.int64,name="input_lr")
        self.x = tf.placeholder(tf.int32,name="input_x")
        self.c = tf.placeholder(tf.int32,name="input_c")
        if tags: self.t = tf.placeholder(tf.int32,name="input_t")

        # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
        self.istate_fw = tf.placeholder("float", [None, 2*num_hidden])
        self.istate_bw = tf.placeholder("float", [None, 2*num_hidden])
        self.y = tf.placeholder(tf.int32, shape=[None,sent_max_len], name="labels")

        # hyper parameters
        self.lr = tf.placeholder(tf.float32,shape=[])
        
        # Define weights
        self._weights = {
            # Hidden layer weights => 2*n_hidden because of foward + backward cells
            'w_emb' : tf.Variable(0.2 * tf.random_uniform([voc_dim,emb_dim], minval=-1.0, maxval=1.0, dtype=tf.float32),name='w_emb',trainable=update),
            'c_emb' : random_uniform([3,emb_dim],'c_emb')
            }
        if tags:
            self._weights.update({'t_emb' : tf.Variable(0.2 * tf.random_uniform([tag_voc_dim,emb_dim], minval=-1.0, maxval=1.0, dtype=tf.float32),name='t_emb',trainable=update)})
        else:
            self._weights = {
                'w_emb' : random_uniform([voc_dim, emb_dim],'w_emb'),
                'c_emb' : random_uniform([3,emb_dim],'c_emb')
                }
            if tags:
                self._weights.update({'t_emb' : random_uniform([tag_voc_dim,emb_dim],'t_emb')})

        self._weights.update({
            'hidden_w': tf.Variable(tf.random_normal([emb_dim, 2*num_hidden])),
            'hidden_c': tf.Variable(tf.random_normal([emb_dim, 2*num_hidden])),
            'out_w': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
                })
        if tags:
                self._weights.update({'hidden_t': tf.Variable(tf.random_normal([emb_dim, 2*num_hidden]))})

        self._biases = {
            'hidden_b': tf.Variable(tf.random_normal([2*num_hidden])),
            'out_b': tf.Variable(tf.random_normal([num_classes]))
        }

        emb_x = tf.nn.embedding_lookup(self._weights['w_emb'],self.x)
        emb_c = tf.nn.embedding_lookup(self._weights['c_emb'],self.c)
        emb_t = tf.nn.embedding_lookup(self._weights['t_emb'],self.t)
        # Linear activation
        _X = tf.matmul(emb_x, self._weights['hidden_w']) + tf.matmul(emb_c, self._weights['hidden_c']) + tf.matmul(emb_t,self._weights['hidden_t']) + self._biases['hidden_b']
        
        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0, state_is_tuple=False)
        
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0,state_is_tuple=False)
        
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        inputs = tf.split(_X, self.sent_max_len, axis=0)
        
        # Get lstm cell output
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs=inputs, 
        sequence_length=self.sequence_lengths, initial_state_fw=self.istate_fw, initial_state_bw=self.istate_bw)
        
        pred = [tf.matmul(tensor,self._weights['out_w'])+ self._biases['out_b'] for tensor in outputs]

        pred = tf.squeeze(tf.stack(pred))

        self.logits = tf.reshape(pred, [-1,self.sent_max_len, self.num_classes])
        
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(inputs=self.logits, tag_indices=self.y, sequence_lengths=self.sequence_lengths)
        
        self.transition_params = transition_params
        self.loss = tf.reduce_mean(-log_likelihood)
