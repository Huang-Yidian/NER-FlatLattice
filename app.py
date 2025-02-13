import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
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

def start_run():
    
    seed = 114514
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    np.random.seed(seed)
    random.seed(seed)

    if conf.device == "-1":
        device = torch.device( "cpu" )  
    else:
        device_id = torch.cuda.device_count()
        device = torch.device("cuda:" + conf.device if torch.cuda.is_available() else "cpu" )
        if not torch.cuda.is_available():
            device = torch.device( "cpu" )  
            
    
    
    wtree, lattice_tokenizer, bigram_tokenizer, label_tokenizer,lattice_embedding,bigram_embedding =  load_msra_tool(conf)
   
    with torch.no_grad():        
        lattice_embedding = lattice_embedding / (torch.norm(lattice_embedding, dim=1, keepdim=True) + 1e-12)
        bigram_embedding  = bigram_embedding / (torch.norm(bigram_embedding, dim=1, keepdim=True) + 1e-12)
    if conf.model == "Base":
        model = BaseModel(conf,lattice_embedding,bigram_embedding,label_tokenizer.max_id+1).to(device) 
    if conf.model == "Lattice":
        model = FlatLattice(conf,lattice_embedding,bigram_embedding,label_tokenizer.max_id+1).to(device)

   
    model.load_state_dict(torch.load(conf.save_path))
    
    """
        这里调用模型，以及接受socket数据，数据用tokenizer处理就行。
    """
    return model, wtree, lattice_tokenizer,bigram_tokenizer,label_tokenizer,device
