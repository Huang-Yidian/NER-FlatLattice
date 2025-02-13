import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
import time
import numpy as np
import random
import warnings
import math
# user define
from utils import *
from config import defualt as conf
from model import *
from trainer import Trainer
from dataset import *
from crf import CRF
from metric import *
warnings.filterwarnings("ignore")

def start_train():
    
    # seg logging configs
    config_logger(conf.log_name,logging.DEBUG,logging.DEBUG)

    # set seed for all possible random process.
    seed = 114514
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logging.info("Set seed {:d} for all random process".format(seed))

    # log conf to logfile
    logging_args(conf=conf)

    # choose devices
    if conf.device == "-1":
        device = torch.device( "cpu" )  
    else:
        device_id = torch.cuda.device_count()
        for device_idx in range(device_id):
            logging.info("{:20s}id {:d}  {:10s}".format("Detect GPU device ",device_idx,torch.cuda.get_device_name(int(conf.device))))
        logging.info("{:20s}id {:d}  {:10s}".format("Using GPU device ",int(conf.device),torch.cuda.get_device_name(int(conf.device))))
        device = torch.device("cuda:" + conf.device if torch.cuda.is_available() else "cpu" )
        if not torch.cuda.is_available():
            device = torch.device( "cpu" )  
            logging.warning("No gpu available, using cpu instead!")
    
    # load train and valid dataset
    # then call for torch dataloader
    # this method make data distributed training easier
    logging.info("Loading Dataset and pretrained lattice weight.............................")
    start_time = time.time()
    trainset, testset, lattice_tokenizer, bigram_tokenizer, label_tokenizer, lattice_embedding, bigram_embedding = load_msra_for_train(conf)
    logging.info("Loading Dataset and pretrained lattice weight finished in {:.4f}".format(time.time() - start_time))

    train_set = MSRADataset(trainset)
    valid_set = MSRADataset(testset)
    logging.info("Loading Dataset finished in {:.4f}".format(time.time() - start_time))
    loader_fn = dataloader_fn()
    if conf.model == "Lattice":
        loader_fn.mode = "flat"

    train_loader = DataLoader(train_set, batch_size = conf.batch, num_workers = conf.num_workers, 
                                shuffle=1 , collate_fn = loader_fn.call_fn)
    test_loader = DataLoader(valid_set, batch_size = conf.test_batch, num_workers =  conf.num_workers, 
                                    shuffle=1, collate_fn = loader_fn.call_fn)

    logging.info("Total cost {:.2f} for loading datasets.".format(time.time()-start_time))
    logging.info("Total {:d} data for training".format(train_set.__len__()))
    logging.info("Total {:d} data for validation".format(valid_set.__len__()))

    with torch.no_grad():        
        lattice_embedding = lattice_embedding / (torch.norm(lattice_embedding, dim=1, keepdim=True) + 1e-12)
        bigram_embedding  = bigram_embedding / (torch.norm(bigram_embedding, dim=1, keepdim=True) + 1e-12)
        
    if conf.model == "Base":
        model = BaseModel(conf,lattice_embedding,bigram_embedding,label_tokenizer.max_id+1).to(device) 

    if conf.model == "Lattice":
        
        model = FlatLattice(conf,lattice_embedding,bigram_embedding,label_tokenizer.max_id+1).to(device)

    if conf.model == "BiLSTM":
        crf = CRF(label_tokenizer.max_id+1).to(device)
        model = BiLSTM(conf,lattice_embedding,bigram_embedding,crf).to(device)
        model.weight_norm(lattice_embedding,bigram_embedding)
        model.lattice_embedding.to(device)
        model.bigram_embedding.to(device)
    if conf.model == "Transformer":
        crf = CRF(label_tokenizer.max_id+1).to(device)
        model = Transformer(conf,lattice_embedding,bigram_embedding,crf).to(device)
        model.weight_norm(lattice_embedding,bigram_embedding)
        model.lattice_embedding.to(device)
        model.bigram_embedding.to(device)
    for n,p in model.named_parameters():
        print('{}:{}'.format(n,p.size()))
    with torch.no_grad():
        print('{}init pram{}'.format('*'*15,'*'*15))
        for n,p in model.named_parameters():
            if 'embedding' not in n and 'pos' not in n and 'PE' not in n \
                    and 'bias' not in n and 'crf' not in n and p.dim()>1:
                    nn.init.xavier_uniform_(p)
                    print('xavier uniform init:{}'.format(n))
                    
        print('{}init pram{}'.format('*' * 15, '*' * 15))
    

    logging.info("Using {:s} model.".format(conf.model))
    # optimizer
    
    
    # all_params = model.parameters()
    # emb_param = []
    # for pname, p in model.named_parameters():
    #     if "embedding" in pname:
    #         emb_param += [p]
    # params_id = list(map(id, emb_param)) 
    # other_params = list(filter(lambda p: id(p) not in params_id, all_params))
	
    if conf.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
        logging.info("Using Adam for optimize!")
        
    elif conf.optim == "sgd":
        optimizer = optim.SGD( model.parameters() ,lr=conf.lr, weight_decay=conf.weight_decay, momentum=conf.momentum)
        #optim.SGD([{'params': other_params}, {'params': emb_param, 'lr': 0.1*conf.lr}],lr=conf.lr, weight_decay=conf.weight_decay, momentum=conf.momentum)
        logging.info("Using SGD for optimize!")
    elif conf.optim == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=conf.lr,weight_decay=conf.weight_decay)
        logging.info("Using AdamW for optimize!")
    elif conf.optim == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr=conf.lr)
        logging.info("Using RMSprop for optimize!")

    warm_up = 0.1*conf.epochs
    max_iter = conf.epochs
    T_max = 20
    lr_max = 1e-4
    lr_min = 1e-5
    lambda0 = lambda cur_iter: cur_iter/warm_up if cur_iter < warm_up else 1/(1+ 0.1*(cur_iter)) 
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

    metrics = Metrics(lattice_tokenizer,label_tokenizer)
    trainer = Trainer(model, conf.epochs, optimizer, conf.valid_step, train_loader, test_loader, metrics, conf, device, scheduler)
    trainer.train()

if __name__ == "__main__":
    start_train()
