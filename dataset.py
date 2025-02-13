import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tf
from utils import *

class MSRADataset(Dataset):
    def __init__(self,dataset) -> None:
        """
            dataset has keys: 
                'sentence', 'labels', 'lattice', 'sentence_idx --- the input ids of chars',
                'lattice_idx --- the input ids of lattice', 'lattice_ps --- position_start',
                'lattice_pe --- position_end', 'label_idx --- the training target in ids'
        """
        super().__init__()
        self.char_idx = dataset["sentence_idx"]
        self.lattice_idx = dataset["lattice_idx"]
        self.bigram = dataset["bigram_idx"]
        self.ps = dataset["lattice_ps"]
        self.pe = dataset["lattice_pe"]
        self.label_idx = dataset["label_idx"]
        
    def __getitem__(self, index):
        char = self.char_idx[index]
        lattice = self.lattice_idx[index]
        ps = self.ps[index]
        pe = self.pe[index]
        label = self.label_idx[index]
        bigram = self.bigram[index]
        return char, lattice,bigram, ps, pe, label

    def __len__(self):
        return len(self.char_idx)


class dataloader_fn(object):
    def __init__(self, mode = "hyd") -> None:
        super().__init__()
        self.mode = mode

    def call_fn(self,batch_data):
        """
            char, lattice, ps, pe, label
        """
        if self.mode == "hyd":
            chars = []
            lattices = []
            bigrams = []
            pss = []
            pes = []
            labels = []
            char_len = []
            lattice_len = []
            max_char_len = 1
            max_lattice_len = 1
            for i in range(len(batch_data)):
                if len(batch_data[i][0]) > max_char_len:
                    max_char_len =  len(batch_data[i][0])
                if len(batch_data[i][1]) > max_lattice_len:
                    max_lattice_len = len(batch_data[i][1])
            
            for i in range(len(batch_data)):
                char = batch_data[i][0] + [ 0 for j in range(max_char_len - len( batch_data[i][0]))]
                lattice = batch_data[i][1] + [ 0 for j in range(max_lattice_len - len( batch_data[i][1]))]
                bigram = batch_data[i][2] + [ 0 for j in range(max_char_len - len( batch_data[i][2]))]
                ps = batch_data[i][3] + [ 0 for j in range(max_lattice_len - len( batch_data[i][3]))]
                pe = batch_data[i][4] + [ 0 for j in range(max_lattice_len - len( batch_data[i][4]))]
                label = batch_data[i][5] + [ 0 for j in range(max_char_len - len( batch_data[i][5]))]
                chars.append( char)
                char_len.append(len( batch_data[i][0]))
                bigrams.append(bigram)
                lattices.append( lattice)
                lattice_len.append(len( batch_data[i][1]))
                pss.append(ps)
                pes.append(pe)
                labels.append(label)
            return torch.tensor(chars,dtype=torch.long), torch.tensor(lattices,dtype=torch.long), torch.tensor(bigrams,dtype=torch.long), \
                    torch.tensor(pss,dtype=torch.long), torch.tensor(pes,dtype=torch.long), torch.tensor(labels,dtype=torch.long), (torch.tensor(char_len),torch.tensor(lattice_len))
        elif self.mode == "flat":
            chars = []
            lattices = []
            bigrams = []
            pss = []
            pes = []
            labels = []
            char_len = []
            lattice_len = []
            max_char_len = 1
            max_lattice_len = 1
            max_len = 1
            for i in range(len(batch_data)):
                if len(batch_data[i][0]) > max_char_len:
                    max_char_len =  len(batch_data[i][0])
                if len(batch_data[i][1]) > max_lattice_len:
                    max_lattice_len = len(batch_data[i][1])
                if len(batch_data[i][1]) + len(batch_data[i][0]) > max_len:
                    max_len = len(batch_data[i][1]) + len(batch_data[i][0]) 
            for i in range(len(batch_data)):
                char = batch_data[i][0] +  batch_data[i][1]  + [ 0 for j in range(max_len - len( batch_data[i][0]) - len( batch_data[i][1] ))]
                lattice = [ 0 for j in range(len( batch_data[i][0]) )] +  batch_data[i][1] + [ 0 for j in range(max_len - len( batch_data[i][0]) - len( batch_data[i][1]))]
                bigram = batch_data[i][2] +  [ 0 for j in range(max_char_len - len( batch_data[i][2]))]
                ps = [k for k in range(len( batch_data[i][0]))] + batch_data[i][3] + [ 0 for j in range(max_len - len( batch_data[i][0]) - len( batch_data[i][1]))]
                pe = [k for k in range(len( batch_data[i][0]))] + batch_data[i][4] + [ 0 for j in range(max_len - len( batch_data[i][0]) - len( batch_data[i][1]))]
                label = batch_data[i][5] + [ 0 for j in range(max_char_len - len( batch_data[i][5]) )]
                chars.append( char)
                char_len.append(len( batch_data[i][0]))
                bigrams.append(bigram)
                lattices.append( lattice)
                lattice_len.append(len( batch_data[i][1]))
                pss.append(ps)
                pes.append(pe)
                labels.append(label)        
            
            return torch.tensor(chars,dtype=torch.long), torch.tensor(lattices,dtype=torch.long), torch.tensor(bigrams,dtype=torch.long), \
                    torch.tensor(pss,dtype=torch.long), torch.tensor(pes,dtype=torch.long), torch.tensor(labels,dtype=torch.long), (torch.tensor(char_len),torch.tensor(lattice_len))