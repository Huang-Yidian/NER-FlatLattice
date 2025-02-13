import torch
from tqdm import tqdm
import logging
import time
import os
import torch.nn as nn

class Trainer(object):
    def __init__(self, model, epochs, optimizer, valid_step, train_loader, test_loader, metric, conf, device,scheduler):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.valid_step = valid_step
        self.train_loader = train_loader
        self.valid_loader = test_loader
        self.test_loader = test_loader
        self.metric = metric
        self.conf = conf
        self.device = device
        self.scheduler = scheduler
        self.print_every = conf.print_every
    
    def train(self ):
        epoch = 1
        best_perform = 0
        if self.conf.use_pretrained and os.path.exists(self.conf.save_path):
            self.model.load_state_dict(torch.load(self.conf.save_path))
        while(epoch < self.epochs + 1):
            self.model.train()
            bar = tqdm(self.train_loader, ncols=90, mininterval=self.print_every)
            train_time = time.time()
            avg_loss = 0
            self.scheduler.step()
            for i, ( char, lattice, bigram, ps, pe, label, lenth) in enumerate(bar):
                bar.set_description(f'Epoch [{epoch}/{self.epochs}]')
                self.optimizer.zero_grad()
                outputs = self.model((char.to(self.device), lattice.to(self.device),bigram.to(self.device), ps.to(self.device), 
                                         pe.to(self.device),label.to(self.device), lenth[0].to(self.device),lenth[1].to(self.device)), is_train=True)
                loss = outputs["loss"]
                avg_loss = loss.item()/(i+1) + avg_loss*i/(i+1) 
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.conf.clip_grad)
                self.optimizer.step() 
                bar.set_postfix(loss = loss.item(), avg_loss = avg_loss)
                if i % self.valid_step == 0 and i != 0 and self.valid_step > 0:
                    best_perform = self.valid(epoch, i, best_perform)
                    self.model.train()

            total_time = time.time() - train_time
            if total_time / 3600 >= 1:
                logging.info("Epoch {:d} finished in {:4f} hours.".format(epoch, total_time / 3600))
            elif total_time / 60 >= 1:
                logging.info("Epoch {:d} finished in {:4f} minutes.".format(epoch, total_time / 60))
            else:
                logging.info("Epoch {:d} finished in {:4f} seconds.".format(epoch, total_time))
            epoch += 1
            best_perform = self.test(epoch,best_perform)   

    def valid(self, epoch, step, best_perform):
        self.model.eval()
        valid_time = time.time()
        with torch.no_grad():
            logging.info("{:20s}".format("Starting Validation!"))
            for i, ( char, lattice, bigram, ps, pe, label, lenth) in enumerate(tqdm(self.valid_loader, ncols=0)):
                outputs = self.model((char.to(self.device), lattice.to(self.device),bigram.to(self.device), ps.to(self.device), 
                            pe.to(self.device),label.to(self.device), lenth[0].to(self.device),lenth[1].to(self.device)), is_train=False)

                self.metric.evaluate(outputs["pred"], label, lenth)
            evaluate_result = self.metric.get_metric(True)
            f, pre, rec = evaluate_result["f"],evaluate_result["pre"],evaluate_result["rec"]
            logging.info("{:30s}{:4f}".format("Current test Fscore is:", f))
            logging.info("{:30s}{:4f}".format("Current test Precision is:", pre))
            logging.info("{:30s}{:4f}".format("Current test Recall is:", rec))
            if  f > best_perform:
                best_perform = f
                torch.save(self.model.state_dict(),self.conf.save_path)
                logging.info("Saving Best model in epoch {:d} step {:d}".format(epoch, step))
        total_time = time.time() - valid_time
        if total_time / 3600 >= 1:
            logging.info("Validation finished in {:4f} hours.".format(total_time / 3600))    
        elif total_time / 60 >= 1:
            logging.info("Validation finished in {:4f} minutes.".format(total_time / 60))    
        else:
            logging.info("Validation finished in {:4f} seconds.".format(total_time))    
        
        return best_perform
    
    def test(self, epoch, best_perform):
        self.model.eval()
        test_time = time.time()
        with torch.no_grad():
            logging.info("{:20s}".format("Starting Test!"))
            for i, ( char, lattice,bigram, ps, pe, label, lenth) in enumerate(tqdm(self.valid_loader, ncols=0)):
                outputs = self.model((char.to(self.device), lattice.to(self.device),bigram.to(self.device), ps.to(self.device), 
                            pe.to(self.device),label.to(self.device), lenth[0].to(self.device),lenth[1].to(self.device)), is_train=False)

                self.metric.evaluate(outputs["pred"], label, lenth[0])
            evaluate_result = self.metric.get_metric(True)
            f, pre, rec = evaluate_result["f"],evaluate_result["pre"],evaluate_result["rec"]
            logging.info("{:30s}{:4f}".format("Current test Fscore is:", f))
            logging.info("{:30s}{:4f}".format("Current test Precision is:", pre))
            logging.info("{:30s}{:4f}".format("Current test Recall is:", rec))
            if  f > best_perform:
                best_perform = f
                torch.save(self.model.state_dict(),self.conf.save_path)
                logging.info("Saving Best model in epoch {:d}.".format(epoch))

        total_time = time.time() - test_time
        if total_time / 3600 >= 1:
            logging.info("Testing finished in {:4f} hours.".format(total_time / 3600))    
        elif total_time / 60 >= 1:
            logging.info("Testing finished in {:4f} minutes.".format(total_time / 60))    
        else:
            logging.info("Testing finished in {:4f} seconds.".format(total_time))    
        return best_perform
