#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import codecs, functools, os, sys
import numpy as np
import cPickle

from itertools import chain
from collections import Counter
from sklearn import metrics
from string import punctuation

# ConLL2obj
# ==================================================
_cue_markers = '<>'
_scope_markers= '{}'

_bracket_escapes = {
    '(': '-LRB-',
    '{': '-LCB-',
    '[': '-LSB-',
    ']': '-RSB-',
    '}': '-RCB-',
    ')': '-RRB-'}

def ptb_escape_brackets(string):
    if string in _bracket_escapes:
        return _bracket_escapes[string]
    else:
        return string

class Data(list):
    def __init__(self, path=None, istream=None):
        super(Data, self).__init__()

        if path:
            self.path = path
            istream = codecs.open(path,'rb','utf8')
        elif istream:
            self.path = None
        else:
            raise ValueError('neither path nor istream specified')

        conll_sentence = []
        for line in istream:
            line = line.strip()
            # collect conll tokens until the entire
            # sentence has been read from file
            if line:
                conll_sentence.append(line.split("\t"))
            else:
                self.append(Sentence(conll_sentence))
                conll_sentence = []
        istream.close()
        # add the final sentence if file ended without blank line
        if conll_sentence:
            self.append(Sentence(conll_sentence))

        self.folds = Folds(self)

    def to_conll(self):
        return '\n\n'.join([sentence.to_conll() for sentence in self])

    def find(self, number):
        for sentence in self:
            if sentence.number == number:
                return sentence

    def correct(self,corrections,deletions,flag):
        # get the sent index we are interested in correcting
        for i in corrections.iterkeys():
            ann_dict = corrections[i]
            # iterate through each token
            # check the annotation 
            for t_i in range(len(self[i])):
                token = self[i][t_i]
                for a_i in range(len(token.get_annotations())):
                    if a_i in ann_dict.keys():
                        if flag=="cue":
                            token.annotations[a_i].cue = None
                        if t_i in ann_dict[a_i].keys():
                            # get word
                            w = ann_dict[a_i][t_i]
                            # reset annotation
                            if flag=="cue":
                                token.annotations[a_i].cue = w
        # delete those spurious annotations
        for i in deletions.iterkeys():
            ann_del = deletions[i]
            for t_i in self[i]:
                new_annotations = []
                for j in range(len(t_i.annotations)):
                    if j not in ann_del:
                        new_annotations.append(t_i.annotations[j])
                t_i.annotations = new_annotations

class Sentence(list):

    def __init__(self, conll_sentence):
        super(Sentence, self).__init__()
        self.number = int(conll_sentence[0][0])
        for conll_token in conll_sentence:
            self.append(Token(self, conll_token))
        self.num_annotations = len(self[0].annotations) if self[0].annotations else 0
        self._tree = None

    def __str__(self):
        return ' '.join([unicode(token) for token in self])

    def sent2tokens(self):
        return [unicode(token) for token in self]

    def __hash__(self):
        return  self.identifier().__hash__()

    def _get_spans(self, annotation_id, accessor):
        spans = []
        start = None
        for token in self:
            annotation = accessor(token.annotations[annotation_id])
            if start == None and annotation:
                start = token.position
            elif start != None and not annotation:
                spans.append((start, token.position - 1))
                start = None
        if start != None: spans.append((start, token.position))
        return spans

    def identifier(self):
        return '%d' % (self.number)

    def get_num_annotations(self):
        return self.num_annotations

    def get_all_cues(self):
        return [[cs for cs in self.get_cues(a)] for a in range(self.num_annotations)]

    def get_cues(self, annotation_id):
        return self._get_spans(annotation_id, Annotation.cue_accessor)

    def get_scopes(self, annotation_id):
        return self._get_spans(annotation_id, Annotation.scope_accessor)

    def get_full_scope(self):
        neg_instances = [[]]*self.num_annotations
        for tok in self:
            anns = tok.annotations
            for x in range(len(tok.annotations)):
                if anns[x].get_cue()!=None:
                    neg_instances[x].append(anns[x].get_cue())
                if anns[x].get_scope()!=None:
                    neg_instances[x].append(anns[x].get_scope())
        return neg_instances

    def discontinuous_scope(self, annotation_id):
        cues = self.get_cues(annotation_id)
        scopes = self.get_scopes(annotation_id)
        for i in range(len(scopes) - 1):
            gap = (scopes[i][1] + 1, scopes[i+1][0] - 1)
            if gap not in cues: return True
        return False

    def get_tree(self):
        if not self._tree:
            ptb = self.to_ptb()
            sexpression = SExpression(ptb)
            self._tree = Tree(sexpression)
        return self._tree

    def to_conll(self):
        return '\n'.join([token.to_conll() for token in self])

    def to_ptb(self):
        return ''.join([token.to_ptb() for token in self])

    def pretty(self, annotation_id=None):

        if self.num_annotations == 0:
            return ' '.join([token.word for token in self])

        strings = []

        if annotation_id != None:
            ids = [annotation_id]
        else:
            ids = range(self.num_annotations)

        in_cue = []
        in_scope = []
        for i in range(max(ids)+1):
            in_cue.append(False)
            in_scope.append(False)

        for i in range(len(self)):
            current = self[i]
            following = self[i+1] if i < len(self) - 1 else None

            cue_starts = Frequencies()
            cue_ends = Frequencies()
            scope_starts = Frequencies()
            scope_ends = Frequencies()


            for identifier in ids:
                in_cue[identifier] = \
                    Token._indices(in_cue[identifier], current, following,
                                    identifier, Annotation.cue_accessor,
                                    cue_starts, cue_ends)
                in_scope[identifier]  = \
                    Token._indices(in_scope[identifier], current, following,
                                    identifier, Annotation.scope_accessor,
                                    scope_starts, scope_ends)

            string = ''
            for i, char in enumerate(self[i].word):

                for j in range(scope_starts[i]): string += _scope_markers[0]
                for j in range(cue_starts[i]): string += _cue_markers[0]
                string += char
                for j in range(cue_ends[i]): string += _cue_markers[1]
                for j in range(scope_ends[i]): string += _scope_markers[1]
            strings.append(string)

        return ' '.join(strings)

    """Added method"""
    def set_new_num_anns(self,new_num_annotations):
        self.num_annotations = new_num_annotations        

    def unravel_neg_instance(self,trg_neg_instances):
        def decide(k,start):
            if k=='cue': ret = start+0
            elif k=='scope': ret = start+2
            return ret
        def substitute(indices,word,ann_string):
            for i in indices:
                ann_string[i] = word
            return ann_string
        start = 0
        dict_tokens = dict()
        print ("TNI: ",len(trg_neg_instances))
        for instance in trg_neg_instances:
            instance_dict = instance.get_elementsAsDict()
            for k in instance_dict:
                for n in instance_dict[k]:
                    dict_tokens.setdefault(n,[]).append(decide(k,start))
            start+=3
        for t in self:
            ann_string = ["_","_","_"] * len(trg_neg_instances)
            pos = t.get_position()
            if pos in dict_tokens:
                word = t.get_word()
                ann_string = substitute(dict_tokens[pos],word,ann_string)
            t.set_annotations(ann_string)

class Token(object):

    _no_annotation = '***'

    accessors = {
        'position': lambda token : token.cue,
        'word': lambda token : token.word,
        'lemma': lambda token : token.lemma
    }


    def __init__(self, sentence, fields):
        self.sentence = sentence
        self.position = int(fields[1])
        self.word = fields[2]
        self.lemma = fields[3]
        self.pos = fields[4]

        if fields[5] == Token._no_annotation:
            self.annotations = []
        else:
            self.annotations = [(Annotation(fields[i:i+2]))for i in range(5, len(fields), 2)]

    def __str__(self):
        return self.word

    def is_punctuation(self):
        return self.word[0] in punctuation

    def to_conll(self):
        fields = [str(self.sentence.number),
                  str(self.position), self.word, self.lemma, self.pos, self.syntax]
        if self.annotations:
            for annotation in self.annotations:
                for field in annotation.fields():
                    fields.append(field)
        else:
            fields.append(Token._no_annotation)
        return '\t'.join(fields)

    # Changed from (pos form) to (pos lemma) -- shouldn't cause a problem...?
    def to_ptb(self):
        return self.syntax.replace('*', '(%s %s)' % (ptb_escape_brackets(self.pos),ptb_escape_brackets(self.lemma)))

    @staticmethod
    def _indices(in_span, current, following, identifier, accessor, starts, ends):
        current_annotation = accessor(current.annotations[identifier])
        start = None
        if current_annotation:
            if not in_span:
                start = current.word.find(current_annotation)
                starts[start] += 1
                in_span = True
            if in_span:
                end_of_span = False

                if len(current_annotation) < len(current.word) and \
                        current.word.find(current_annotation) != \
                        len(current.word) - len(current_annotation):
                    end_of_span = True
                elif not following:
                    end_of_span = True
                else:
                    following_annotation = accessor(following.annotations[identifier])
                    if not following_annotation:
                        end_of_span = True
                    else:
                        if current.word.find(current_annotation) + len(current_annotation) < len(current.word) - 1:
                            end_of_span = True
                        elif following_annotation != following.word and \
                                following.word.find(following_annotation) > 0:
                            end_of_span = True

                if end_of_span:
                    end = len(current_annotation) - 1
                    if start: end += start
                    ends[end] += 1
                    in_span = False

        return in_span

    def get_word(self):
        return self.word

    def get_position(self):
        return self.position

    def get_annotations(self):
        return self.annotations

    def set_annotations(self,fields):
        self.annotations = [(Annotation(fields[i:i+2]))for i in range(0,len(fields), 2)]

    def is_cue(self):
        cues = [a.get_cue() for a in self.get_annotations()]
        return 0 if all(x==None for x in cues) else 1

    def is_scope(self):
        scope_els = [a.get_scope() for a in self.get_annotations()]
        return 0 if all(x==None for x in scope_els) else 1

class Annotation(object):

    _null = '_'

    cue_accessor = lambda annotation : annotation.cue
    scope_accessor = lambda annotation : annotation.scope

    def __init__(self, fields):
        self.cue = Annotation._in(fields[0])
        self.scope = Annotation._in(fields[1])

    def get_cue(self):
        return self.cue

    def get_scope(self):
        return self.scope

    def get_elements_tuple(self):
        return (self.cue,self.scope)

    def fields(self):
        return [Annotation._out(field) for field in
                (self.cue, self.scope)]

    @staticmethod
    def _in(field):
        return None if field == Annotation._null else field

    @staticmethod
    def _out(field):
        return field if field else Annotation._null

# Common
# ==================================================
class Folds(list):
    def __init__(self, items, n=10):
        super(Folds, self).__init__()
        for i in range(n): self.append(set())
        self.xref = dict()
        fold = 0
        for item in items:
            self[fold].add(item)
            self.xref[item.identifier()] = fold
            fold += 1
            if fold == len(self): fold = 0  
            
    def get_fold(self, item):
        return self.xref[item.identifier()]
            
class Frequencies(dict):
    def __getitem__(self, key):
        return super(Frequencies, self).__getitem__(key) if key in self else 0

    def __setitem__(self, key, value):
        assert value.__class__ == int
        return super(Frequencies, self).__setitem__(key, value)

    def total(self):
        return sum(self.values())

# Utils
# ==================================================
def data2sents(sets,look_scope,lang):
    def get_uni_mapping(lang):
        mapping = {}
        f = codecs.open('../%s.txt' % lang,'rb','utf8').readlines()
        for line in f:
            spl = line.strip().split('\t')
            _pos = spl[0].split('|')[0]
            mapping.update({_pos:spl[1]})
        return mapping
    
    def segment(word,is_cue):
		
        if is_cue:
            return ([word],None)

        else:
            return ([word],None)

    def assign_tag(is_scope,look_scope):
        if is_scope and look_scope:
            return 'I'
        else: return 'O'

    sents = []
    tag_sents = []
    ys = []
    lengths = []

    cues_one_hot = []
    scopes_one_hot = []

    for d in sets:
        length = 0
        for s_idx,s in enumerate(d):
            all_cues = [i for i in range(len(s)) if filter(lambda x: x.cue!=None,s[i].annotations)!=[]]
            if len(s[0].annotations) > 0:
                for curr_ann in range(len(s[0].annotations)):
                    cues_idxs = [i[0] for i in filter(lambda x: x[1]!=None,[(i,s[i].annotations[curr_ann].cue) for i in range(len(s))])]
                    scope_idxs = [i[0] for i in filter(lambda x: x[1]!=None,[(i,s[i].annotations[curr_ann].scope) for i in range(len(s))])]

                    sent = []
                    tag_sent = []
                    y = []

                    cue_one_hot = []
                    scope_one_hot = []

                    for t_idx,t in enumerate(s):
                        word,tag = t.word,t.pos
                        word_spl,word_idx = segment(word, t_idx in all_cues)
                        if len(word_spl) == 1:
                            _y = assign_tag(t_idx in scope_idxs,look_scope)
                            c_info = ['NOTCUE'] if t_idx not in cues_idxs else ["CUE"]
                            s_info = ['S'] if t_idx in scope_idxs else ['NS']
                            tag_info = [tag]

                        elif len(word_spl) == 2:
                            _y_word = assign_tag(t_idx in scope_idxs,look_scope)
                            if t_idx in cues_idxs:
                                _y = [_y_word,'O'] if word_idx == 0 else ['O',_y_word]
                                c_info = ['NOTCUE','CUE'] if word_idx == 0 else ['CUE','NOTCUE']
                                s_info = ['S','NS'] if word_idx == 0 else ['NS','S']
                            else:
                                _y = [_y_word,_y_word]
                                c_info = ['NOTCUE','NOTCUE']
                                s_info = ['S','S'] if t_idx in scope_idxs else ['NS',"NS"]
                            tag_info = [tag,'AFF'] if word_idx == 0 else ['AFF',tag]
                        # add the word(s) to the sentence list
                        sent.extend(word_spl)
                        # add the POS tag(s) to the TAG sentence list
                        tag_sent.extend(tag_info)
                        # add the _y for the word
                        y.extend(_y)
                        # extend the cue hot vector
                        cue_one_hot.extend(c_info)
                        # extend the scope hot vector
                        scope_one_hot.extend(s_info)

                    sents.append(sent)
                    tag_sents.append(tag_sent)
                    print tag_sents
                    ys.append(y)
                    cues_one_hot.append(cue_one_hot)
                    scopes_one_hot.append(scope_one_hot)
                    length+=1

        lengths.append(length)
    # make normal POS tag into uni POS tags
    pos2uni = get_uni_mapping(lang)
    tag_uni_sents = [[pos2uni[t] for t in _s] for _s in tag_sents]

    return sents,tag_sents,tag_uni_sents,ys,cues_one_hot,scopes_one_hot,lengths

# INT Processor
# ==================================================
fn_training = os.path.abspath('training set')
fn_dev = os.path.abspath('validation set')

def load_train_dev(scope,lang,out_dir):
    # read data,get sentences as list of lists
    training = Data(fn_training)
    dev = Data(fn_dev)

    # get all strings
    sents,tags,tags_uni,labels,cues,scopes,lengths = data2sents([training,dev],scope,lang)

    # build vocabularies
    voc, voc_inv = build_vocab(sents, tags, tags_uni,labels,lengths)

    # transform the tokens into integer indices
    words_idxs, tags_idxs, tags_uni_idxs, cues_idxs, scopes_idxs, labels_idxs = build_input_data(voc, sents, tags, tags_uni, cues, scopes, labels)

    # package
    data = package_data_train_dev(words_idxs, tags_idxs, tags_uni_idxs, cues_idxs, scopes_idxs, labels_idxs, voc, voc_inv, lengths)

    # pickle data
    pickle_data(out_dir, data)

    return data

def load_test(fn_test,voc,scope,lang):
    # get test set
    test = reduce(lambda x,y:x+y,map(lambda z: Data(z),fn_test))

    # process the test set
    sents,tags,tags_uni,labels,cues,scopes,_ = data2sents([test],scope,lang)

    # transform the tokens into integer indices
    words_idxs, tags_idxs, tags_uni_idxs, cues_idxs, scopes_idxs, labels_idxs = build_input_data(voc, sents, tags, tags_uni, cues, scopes, labels)

    return words_idxs, tags_idxs, tags_uni_idxs, cues_idxs, scopes_idxs, labels_idxs

def build_vocab(sents, tags, tags_uni, labels,lengths):
    def token2idx(cnt):
        return dict([(w,i) for i,w in enumerate(cnt.keys())])

    w2idxs = token2idx(Counter(chain(*sents)))
    # add <UNK> token
    w2idxs['<UNK>'] = max(w2idxs.values())+1
    t2idxs = token2idx(Counter(chain(*tags)))
    tuni2idxs= token2idx(Counter(chain(*tags_uni)))
    y2idxs = {'I':0,'O':1,'E':2}

    voc,voc_inv = {},{}
    voc['w2idxs'],voc_inv['idxs2w'] = w2idxs, {i: x for x,i in w2idxs.iteritems()}
    voc['y2idxs'],voc_inv['idxs2y'] = y2idxs, {i: x for x,i in y2idxs.iteritems()}
    voc['t2idxs'],voc_inv['idxs2t'] = t2idxs, {i: x for x,i in t2idxs.iteritems()}
    voc['tuni2idxs'],voc_inv['idxs2tuni'] = tuni2idxs, {x: i for x,i in tuni2idxs.iteritems()}
 
    return voc,voc_inv

def build_input_data(voc, sents, tags, tags_uni, cues, scopes, labels):

    words_idxs = [np.array([voc['w2idxs'][w] if w in voc['w2idxs'] else voc['w2idxs']["<UNK>"] for w in sent],dtype=np.int32) for sent in sents]

    tags_idxs = [np.array([voc['t2idxs'][t] for t in tag_sent],dtype=np.int32) for tag_sent in tags]
    tags_uni_idxs = [np.array([voc['tuni2idxs'][tu] for tu in tag_sent_uni],dtype=np.int32) for tag_sent_uni in tags_uni]
    y_idxs = [np.array([voc['y2idxs'][y] for y in y_array],dtype=np.int32) for y_array in labels]
    cues_idxs = [np.array([1 if c=="CUE" else 0 for c in c_array],dtype=np.int32) for c_array in cues]
    scope_idxs = [np.array([1 if s=="S" else 0 for s in s_array],dtype=np.int32) for s_array in scopes]

    return words_idxs, tags_idxs, tags_uni_idxs, cues_idxs, scope_idxs,  y_idxs

def package_data_train_dev(sent_ind_x,tag_ind_x,tag_uni_ind_x,sent_ind_y,cues_idxs,scopes_idxs,voc,voc_inv,lengths):

    # vectors of words
    train_x, dev_x = sent_ind_x[:lengths[0]],sent_ind_x[lengths[0]:lengths[0]+lengths[1]]

    # vectors of POS tags
    train_tag_x, dev_tag_x = tag_ind_x[:lengths[0]],tag_ind_x[lengths[0]:lengths[0]+lengths[1]]

    # vectors of uni POS tags
    train_tag_uni_x, dev_tag_uni_x = tag_uni_ind_x[:lengths[0]],tag_uni_ind_x[lengths[0]:lengths[0]+lengths[1]]

    # vectors of y labels
    train_y, dev_y= sent_ind_y[:lengths[0]],sent_ind_y[lengths[0]:lengths[0]+lengths[1]]

    # vectors of cue info
    train_cue_info,dev_cue_info = cues_idxs[:lengths[0]],cues_idxs[lengths[0]:lengths[0]+lengths[1]]

    # vectors of scope info
    train_scope_info, dev_scope_info = scopes_idxs[:lengths[0]],scopes_idxs[lengths[0]:lengths[0]+lengths[1]]

    train_set = [train_x, train_tag_x, train_tag_uni_x, train_y, train_cue_info, train_scope_info]
    dev_set = [dev_x, dev_tag_x, dev_tag_uni_x, dev_y, dev_cue_info, dev_scope_info]

    return [train_set,dev_set, voc, voc_inv]

def pickle_data(out_dir,data):

    print ("Storing data to %s..." % out_dir)
    with open(os.path.join(out_dir,'train_dev.pkl'),'wb') as f:
        cPickle.dump(data,f)
    print ("Data stored!")

   
# METRICS
# ==================================================
def get_eval(predictions,gs):
    y,y_ = [],[]
    for p in predictions: y.extend(map(lambda x: 0 if list(x)==[0] else 1,p))
    for g in gs: y_.extend(map(lambda x: 0 if list(x)==[0] else 1,g))
    print (metrics.classification_report(y_,y))
    cm = metrics.confusion_matrix(y_,y)
    print (cm)

    p,r,f1,s =  metrics.precision_recall_fscore_support(y_,y)
    report = "%s\n%s\n%s\n%s\n\n" % (str(p),str(r),str(f1),str(s)) 

    f1_pos = f1[0]

    return np.average(f1,weights=s),report,cm,f1_pos

def write_report(folder, report, cm, name):
    print ("Storing reports...")           
    with codecs.open(os.path.join(folder,'%s_report.txt' % name),'wb','utf8') as store_rep_dev:
        store_rep_dev.write(report)
        store_rep_dev.write(str(cm)+"\n")
    print ("Reports stored...")

def store_prediction(folder, lex, dic_inv, pred_dev, gold_dev, name):
    print ("Storing labelling results for dev or test set...")
    with codecs.open(os.path.join(folder,'best_%s.txt' % name),'wb','utf8') as store_pred:
        for s, y_sys, y_hat in zip(lex,pred_dev,gold_dev):
            s = [dic_inv['idxs2w'][w] if w in dic_inv['idxs2w'] else '<UNK>' for w in s]
            assert len(s)==len(y_sys)==len(y_hat)
            for _word,_sys,gold in zip(s,y_sys,y_hat):
                _p = 0 if list(_sys)==[0] else 1
                _g = 0 if list(gold)==[0] else 1
                store_pred.write("%s\t%s\t%s\n" % (_word,_g,_p))
            store_pred.write("\n")
