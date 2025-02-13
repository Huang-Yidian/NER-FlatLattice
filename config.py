import easydict
import time
defualt = easydict.EasyDict()
defualt.train_path =  "./MSRA/train_dev.char.bmes_clip2"
defualt.test_path = "./MSRA/test.char.bmes_clip2"
defualt.char_emb = "./weight/gigaword_chn.all.a2b.uni.ite50.vec"
defualt.bi_emb = "./weight/gigaword_chn.all.a2b.bi.ite50.vec"
defualt.word_emb = "./weight/ctb.50d.vec"
defualt.emb_path = "./weight/all.vec"

defualt.batch = 4
defualt.test_batch = 4
defualt.print_every = 1
defualt.model = "Lattice"  #    "Base"  "Lattice" "BiLSTM" "Transformer"
defualt.max_seq_len = 256
defualt.emb_size = 50
defualt.hidden_size = 160
defualt.num_head = 8

# conv_layer conv, char_layer char, lattice_layer lattice and a pair fuse network in a block
defualt.char_layer = 1
defualt.lattice_layer = 0
defualt.char_lattice_layer = 1
defualt.lattice_char_layer = 0

if defualt.model == "Transformer":
    defualt.Transformer = 2
    
defualt.abs_pos = True
defualt.rel_pos = True

defualt.device = "0"

defualt.epochs = 100
defualt.optim = "sgd"             #rms|sgd|adam|adamw
defualt.lr = 1e-3 
defualt.weight_decay = 1e-4
defualt.momentum = 0.9
defualt.save_path = "./pretrained.pt"
defualt.use_pretrained = 0

defualt.log_name = "./log/" +  time.strftime("%Y%m%d%H%M%S", time.localtime())  + ".txt"

defualt.num_workers = 0

defualt.valid_step = -1#20000

defualt.drop_ff = 0.0
defualt.drop_emb = 0.5
defualt.drop_attn = 0.0
defualt.drop_ff1=0.15
defualt.drop_ff2=0.0
defualt.drop_output=0.3



defualt.clip_grad = 5
 