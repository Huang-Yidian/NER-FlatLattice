import logging
import sys
import numpy as np
import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import hashlib
import os
import pickle

cache_root_dir = 'cache'
if not os.path.exists(cache_root_dir):
    os.makedirs(cache_root_dir)
def md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()
def cache_key(f, *args, **kwargs):
    s = '%s-%s-%s' % (f.__name__, str(args), str(kwargs))
    return os.path.join(cache_root_dir, '%s.dump' % md5(s.encode("utf-8")))
def cache(f):
    def wrap(*args, **kwargs):
        fn = cache_key(f, *args, **kwargs)
        if os.path.exists(fn):
            print('loading cache')
            with open(fn, 'rb') as fr:
                return pickle.load(fr)
        obj = f(*args, **kwargs)
        with open(fn, 'wb') as fw:
            pickle.dump(obj, fw)
        return obj
    return wrap


def logging_args(conf):
    for key in conf.keys():
        logging.info("{:20s}{:10s}".format(key,str(conf[key])))


def config_logger(file_name, file_level,console_level):
    file_handler = logging.FileHandler(file_name, mode='w', encoding="utf8")
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    file_handler.setLevel(file_level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    console_handler.setLevel(console_level)
    logging.basicConfig(level = min(console_level, file_level), 
                        handlers = [file_handler, console_handler])


@cache
def load_lattice(emb_path):
    lattice = []
    emb = []
    with open(emb_path,"r",encoding = "utf-8") as f :
        for line in f:
            lattice_emb = line.strip().strip("\n").split()
            lattice.append(lattice_emb[0])
            emb.append(list(map(float, lattice_emb[1:])))
    return lattice, emb

@cache
def load_char(emb_path):
    lattice = []
    emb = []
    with open(emb_path,"r",encoding = "utf-8") as f :
        for line in f:
            lattice_emb = line.strip().strip("\n").split()
            lattice.append(lattice_emb[0])
            emb.append(list(map(float, lattice_emb[1:])))
    return lattice, emb

@cache
def load_word(emb_path):
    words = []
    emb = []
    with open(emb_path,"r",encoding = "utf-8") as f :
        for line in f:
            words_emb = line.strip().strip("\n").split()
            words.append(words_emb[0])
            emb.append(list(map(float, words_emb[1:])))
    return words, emb
@cache
def load_bigram(emb_path):
    bigram = []
    emb = []
    with open(emb_path,"r",encoding = "utf-8") as f :
        for line in f:
            bigram_emb = line.strip().strip("\n").split()
            bigram.append(bigram_emb[0])
            emb.append(list(map(float, bigram_emb[1:])))
    return bigram, emb  

def concate_word_emb(conf):
    dic = {}
    target = open(conf.emb_path,"w",encoding="utf-8")
    with open(conf.char_emb,"r",encoding="utf-8") as f: 
            for line in f.readlines():
                char = line.split(" ")[0]
                if char not in dic.keys():
                    dic[char] = 1
                    target.write(line)
    # with open(conf.bi_emb,"r",encoding="utf-8") as f:
    #          for line in f.readlines():
    #             char = line.split(" ")[0]
    #             if char not in dic.keys():
    #                 dic[char] = 1
    #                 target.write(line)
    with open(conf.word_emb,"r",encoding="utf-8") as f:
             for line in f.readlines():
                char = line.split(" ")[0]
                if char not in dic.keys() and len(char) > 1:
                    dic[char] = 1
                    target.write(line)
import random
class Tokenizer(object):
    """_summary_
    0 is pad
    1 is unknown
    Args:
        object (_type_): _description_
    """
    def __init__(self,lat, idx, emb) -> None:
        self.token_dict = {}
        self.idx_dict = {}
        self.emb = [[ random.uniform(-1,1) for i in range(len(emb[0]))] , [random.uniform(-1,1) for i in range(len(emb[0]))]] + emb
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.token_dict[self.pad] = [0, 1]
        self.idx_dict[0] = [self.pad]
        self.token_dict[self.unk] = [1, 1]
        self.idx_dict[1] = [self.unk]
        for lattice,index in zip(lat, idx):
            self.token_dict[lattice] = [index+2,0]
            self.idx_dict[index+2] = [lattice]
        self.max_id = len(self.token_dict.keys())-1
        
    def tokenize_char(self, dataset, mode = "add"):
        sentence = dataset["sentence"]
        lattice = dataset["lattice"]
        sentence_idx = []
        max_seqlen = 0
        for seq in sentence:
        
            if len(seq)>max_seqlen:
                max_seqlen = len(seq)
            seq_idx = []

            for char in seq:
                if char in  self.token_dict.keys():
                    self.token_dict[char][1] = 1
                    seq_idx.append(self.token_dict[char][0] )
                if char not in self.token_dict.keys() and mode == "add":
                    self.token_dict[char] = [self.max_id+1, 1]
                    self.idx_dict[self.max_id+1] = [char]
                    self.max_id += 1
                    seq_idx.append(self.token_dict[char][0] )
                    self.emb.append([ random.uniform(-1,1) for i in range(len(self.emb[0]))])
                if char not in self.token_dict.keys() and mode != "add":
                    seq_idx.append(1)

            sentence_idx.append(seq_idx)

        return sentence_idx
    
    def tokenize_word(self, dataset, mode = "add"):# no add for the word list is predefined
        lattice = dataset["lattice"]

        lattice_idx = []
        lattice_ps = []
        lattice_pe = []
        for item in lattice:
            lat_idx = []
            lat_ps = []
            lat_pe = []
            for words in item:
                s,e,w = words
                lat_ps.append(s)
                lat_pe.append(e)
                if w in  self.token_dict.keys():
                    self.token_dict[w][1] = 1
                    lat_idx.append(self.token_dict[w][0] )
                if w not in self.token_dict.keys() and mode == "add":
                    self.token_dict[w] = [self.max_id+1, 1]
                    self.idx_dict[self.max_id+1] = [w]
                    self.max_id += 1
                    lat_idx.append(self.token_dict[w][0] )
                    self.emb.append([ random.uniform(-1,1) for i in range(len(self.emb[0]))])
                if w not in self.token_dict.keys() and mode != "add":
                    lat_idx.append(1)

           
            lattice_idx.append(lat_idx)
            lattice_ps.append(lat_ps)
            lattice_pe.append(lat_pe)

        return lattice_idx,lattice_ps,lattice_pe
    
    def tokenize_bigram(self, dataset, mode = "add"): 
        bigrams = dataset["bigram"]
        bigram_idx = []
        for bigram in bigrams:
            big_idx = []
            for bi in bigram:
                if bi in self.token_dict.keys():
                    self.token_dict[bi][1] = 1
                    big_idx.append(self.token_dict[bi][0] )
                if bi not in self.token_dict.keys() and mode == "add": 
                    self.token_dict[bi] = [self.max_id+1, 1]
                    self.idx_dict[self.max_id+1] = [bi]
                    self.max_id += 1
                    big_idx.append(self.token_dict[bi][0] )
                    self.emb.append([ random.uniform(-1,1) for i in range(len(self.emb[0]))])
                if bi not in self.token_dict.keys() and mode != "add":
                    big_idx.append(1)

            bigram_idx.append(big_idx)
        return bigram_idx

    def sort(self):
        token = {}
        idx = {}
        emb = []
        cnt = 0
        for lattice,(id,appear) in self.token_dict.items():
            if appear != 0:
                token[lattice] = [cnt, 1]
                idx[cnt] = [lattice]
                emb.append(self.emb[id])
                cnt += 1
        self.token_dict = token
        self.idx_dict = idx
        self.emb = emb
        self.max_id = len(self.token_dict.keys())-1
                
                

class LabelTokenizer(object):

    def __init__(self,labels,idx) -> None:
        self.token_dict = {}
        self.idx_dict = {}
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.token_dict[self.pad] = [0]
        self.idx_dict[0] = self.pad
        self.token_dict[self.unk] = [1]
        self.idx_dict[1] = self.unk
        for label,index in zip(labels,idx):
            self.token_dict[label] = index+2
            self.idx_dict[index+2] = label
        self.max_id = len(self.token_dict.keys()) - 1
        self.labels = [self.pad, self.unk] + labels
        
    def tokenize(self, label, mode = "add"):
        label_idx = []
        for seq_label in label:
            seq_idx = []
            for char_label in seq_label:
                if char_label in  self.token_dict.keys():
                    seq_idx.append(self.token_dict[char_label])
                if char_label not in  self.token_dict.keys() and mode == "add":
                    self.token_dict[label] =  self.max_id + 1
                    self.idx_dict[self.max_id + 1] = label
                    self.max_id += 1
                    seq_idx.append(self.token_dict[char_label])
                if char_label not in  self.token_dict.keys() and mode != "add":
                    seq_idx.append(1)
            label_idx.append(seq_idx)
        return label_idx
    
class Node(object):
    def __init__(self) -> None:
        self.children = collections.defaultdict(Node)
        self.is_word = False

class word_tree(object):
    def __init__(self) -> None:
        self.root = Node()

    def insert_word(self,word):
        current = self.root
        for char in word:
            current = current.children[char]
        current.is_word = True
    
    def match(self, word):
        current = self.root
        for c in word:
            current = current.children.get(c)
            if current is None:
                return -1
        if current.is_word:
            return 1
        else:
            return 0

    def get_words(self,sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break
                if current.is_word:
                    result.append([i,j,sentence[i:j+1]])
        return result
    
def get_words_in_sentence(sentence:list,wtree:word_tree):
    words_list = []
    for i in range(len(sentence)):
        words_list.append(wtree.get_words("".join(sentence[i])))
    return words_list

def get_bigram_in_sentence(sentence:list):
    bigram_list = []
    for i in range(len(sentence)):
        bigram = []
        for j in range(len(sentence[i])):
            if j == 0:
                bigram.append("</s>" + sentence[i][j])
            else:
                bigram.append(sentence[i][j-1] + sentence[i][j])
        bigram_list.append(bigram)
    return bigram_list


def load_emb_tokenizer_wtree(conf):
    lattice, lattice_emb = load_lattice(conf.emb_path)
    # char, char_emb = load_char(conf.char_emb)
    # words, words_emb = load_word(conf.word_emb)
    bigram, bigram_emb = load_bigram(conf.bi_emb)
    logging.info("Load {:d} lattice in {:20s}".format(len(lattice),conf.emb_path))
    logging.info("Load {:d} Bigram in {:20s}".format(len(bigram),conf.bi_emb))
    idx = [i for i in range(len(lattice))]
    lattice_tokenizer = Tokenizer(lattice, idx, lattice_emb)

    idx = [i for i in range(len(bigram))]
    bigram_tokenizer = Tokenizer(bigram, idx, bigram_emb)
    wtree = word_tree()
    for word in lattice:
        # make sure ith is a word, not char. (lattice has both char and word)
        if len(word) > 1:
            wtree.insert_word(word)

    return wtree,lattice_tokenizer,bigram_tokenizer

def load_msra(conf):
    label_list = []
    sentence = []
    labels = []
    with open(conf.train_path, "r", encoding="utf-8") as f:
        seq = []
        label = []
        for line in f:
            char_label = line.strip().strip("\n").split(" ")
            if char_label == [""]:
                if len(seq) >= 1:
                    sentence.append(seq)
                    labels.append(label)
                    seq = []
                    label = []
            else:
                seq.append(char_label[0])
                label.append(char_label[1])
                if char_label[1] not in label_list:
                    label_list.append(char_label[1])
    train_set = {"sentence":sentence, "labels":labels}  
    sentence = []
    labels = []
    with open(conf.test_path, "r", encoding="utf-8") as f:
        seq = []
        label = []
        for line in f:
            char_label = line.strip().strip("\n").split(" ")
            if char_label == [""]:
                if len(seq) >= 1:
                    sentence.append(seq)
                    labels.append(label)
                    seq = []
                    label = []
            else:
                seq.append(char_label[0])
                label.append(char_label[1])
                if char_label[1] not in label_list:
                    label_list.append(char_label[1])
    test_set = {"sentence":sentence, "labels":labels}  
    tokenizer = LabelTokenizer(label_list, [i for i in range(len(label_list))])          
    return train_set, test_set, tokenizer

def create_embedding(tokenizer:Tokenizer):
    embedding = torch.tensor(tokenizer.emb)
    return embedding

def load_msra_for_train(conf):
    """
        1 get tree and tokenizer and data
        2 get words in sentence
        3 tokenize all data
    """
    wtree,lattice_tokenizer,bigram_tokenizer = load_emb_tokenizer_wtree(conf)
    trainset, testset, label_tokenizer = load_msra(conf)

    trainset_words = get_words_in_sentence(trainset["sentence"], wtree)
    trainset_bigram = get_bigram_in_sentence(trainset["sentence"])
    trainset["lattice"] = trainset_words
    trainset["bigram"] = trainset_bigram
    
    testset_words = get_words_in_sentence(testset["sentence"],wtree)
    test_bigram = get_bigram_in_sentence(testset["sentence"])
    testset["lattice"] = testset_words
    testset["bigram"] = test_bigram
    
    # keep those have be appeared
    _ = lattice_tokenizer.tokenize_char(trainset,"add")
    _,_,_ =lattice_tokenizer.tokenize_word(trainset,"add")
    _ = bigram_tokenizer.tokenize_bigram(trainset,"add")
    # _ = lattice_tokenizer.tokenize_char(testset,"add")
    # _,_,_ =lattice_tokenizer.tokenize_word(testset,"add")
    # _ = bigram_tokenizer.tokenize_bigram(testset,"add")
    
    lattice_tokenizer.sort()
    logging.info("{:d} lattice rest after sort".format(len(lattice_tokenizer.emb)))
    bigram_tokenizer.sort()
    logging.info("{:d} bigram rest after sort".format(len(bigram_tokenizer.emb)))
    
    sentence_idx = lattice_tokenizer.tokenize_char(trainset,"no add")
    lattice_idx,lattice_ps,lattice_pe =lattice_tokenizer.tokenize_word(trainset,"no add")
    bigram_idx = bigram_tokenizer.tokenize_bigram(trainset,"no add")
    trainset["sentence_idx"] = sentence_idx
    trainset["bigram_idx"] = bigram_idx
    trainset["lattice_idx"] = lattice_idx
    trainset["lattice_ps"] = lattice_ps
    trainset["lattice_pe"] = lattice_pe
    trainset["label_idx"] = label_tokenizer.tokenize(trainset["labels"])
    
    sentence_idx = lattice_tokenizer.tokenize_char(testset,"no add")
    lattice_idx,lattice_ps,lattice_pe =lattice_tokenizer.tokenize_word(testset,"no add")
    bigram_idx = bigram_tokenizer.tokenize_bigram(testset,"no add")
    testset["sentence_idx"] = sentence_idx
    testset["bigram_idx"] = bigram_idx
    testset["lattice_idx"] = lattice_idx
    testset["lattice_ps"] = lattice_ps
    testset["lattice_pe"] = lattice_pe
    testset["label_idx"] = label_tokenizer.tokenize(testset["labels"])

    lattice_embedding = create_embedding(lattice_tokenizer)
    bigram_embedding = create_embedding(bigram_tokenizer)
    return trainset, testset, lattice_tokenizer,bigram_tokenizer, label_tokenizer, lattice_embedding,bigram_embedding

def load_msra_tool(conf):
    wtree,lattice_tokenizer,bigram_tokenizer = load_emb_tokenizer_wtree(conf)
    trainset, testset, label_tokenizer = load_msra(conf)

    trainset_words = get_words_in_sentence(trainset["sentence"], wtree)
    trainset_bigram = get_bigram_in_sentence(trainset["sentence"])
    trainset["lattice"] = trainset_words
    trainset["bigram"] = trainset_bigram
    
    testset_words = get_words_in_sentence(testset["sentence"],wtree)
    test_bigram = get_bigram_in_sentence(testset["sentence"])
    testset["lattice"] = testset_words
    testset["bigram"] = test_bigram
    
    # keep those have be appeared
    _ = lattice_tokenizer.tokenize_char(trainset,"add")
    _,_,_ =lattice_tokenizer.tokenize_word(trainset,"add")
    _ = bigram_tokenizer.tokenize_bigram(trainset,"add")
    # _ = lattice_tokenizer.tokenize_char(testset,"add")
    # _,_,_ =lattice_tokenizer.tokenize_word(testset,"add")
    # _ = bigram_tokenizer.tokenize_bigram(testset,"add")
    
    lattice_tokenizer.sort()
    logging.info("{:d} lattice rest after sort".format(len(lattice_tokenizer.emb)))
    bigram_tokenizer.sort()
    logging.info("{:d} bigram rest after sort".format(len(bigram_tokenizer.emb)))
    lattice_embedding = create_embedding(lattice_tokenizer)
    bigram_embedding = create_embedding(bigram_tokenizer)
    return wtree, lattice_tokenizer, bigram_tokenizer, label_tokenizer,lattice_embedding,bigram_embedding