import numpy as np
import cPickle
import shutil
import collections
import math
from scipy.stats.mstats import gmean
import sys
import ast

import torch
from torch.autograd import Variable

from tqdm import tqdm

from models import *
from dataLoader import dataLoader
from evaluate import evaluate

class Experiment(object):
    def __init__(self, train_data_dir=None, train_meta_file=None, id_map=None,
                valid_data_dir=None, valid_meta_file=None,
                test_data_dir=None, test_meta_file=None,
                label_ignore=None,
                method = MultiMA, n_users=None, n_items=None, n_labels=None, hyperparameters = [1], class_weight=None,
                batch_size = 128,
                lr = 0.01,
                weight_decay = 0.000,
                optimizer = torch.optim.Adam,
                n_epochs = 1000,
                random_seed = None,
                dataLoader = dataLoader,
                num_workers = 0,
                checkpoint_dir = "ckpt/",
                checkpoint_file = "latest.tar",
                bestpoint_file = "best.tar",
                log_file = "logs/",
                resume = None
                ):
        self.train_data_dir = train_data_dir
        self.train_meta_file = train_meta_file
        self.id_map = id_map
        self.valid_data_dir = valid_data_dir
        self.valid_meta_file = valid_meta_file
        self.test_data_dir = test_data_dir
        self.test_meta_file = test_meta_file
        self.label_ignore = label_ignore
        if self.label_ignore is None:
            self.model = method(n_users, n_items, n_labels, hyperparameters)
        else:
            self.model = method(n_users, n_items, n_labels-1, hyperparameters)
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer(self.model.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.weight_decay)
        self.n_epochs = n_epochs
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        self.dataLoader = dataLoader
        self.num_workers = num_workers
        self.checkpoint_file = checkpoint_dir + str(method.__name__) + "_" + "H".join(map(str,hyperparameters)) + "_" + checkpoint_file
        self.bestpoint_file  = checkpoint_dir + str(method.__name__) + "_" + "H".join(map(str,hyperparameters)) + "_" + bestpoint_file
        self.log_file = log_file + str(method.__name__) + "_" + "H".join(map(str,hyperparameters)) + "_" 
        with open(self.log_file, "a") as logf:
            logf.write("weight_decay: %f\n" % weight_decay)
            logf.write("lr: %f\n" % self.lr)
            logf.write("batch_size: %d\n" % self.batch_size)

        self.epoch_start = 1
        self.best_loss = 1000.0
        if resume is not None:
            checkpoint = torch.load(resume)
            self.epoch_start = checkpoint["epoch"] + 1
            self.best_loss = checkpoint["best_loss"]
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print "loaded checkpoint '{}' (epoch {}) with best_loss {}"\
                .format(resume, checkpoint["epoch"], checkpoint["best_loss"])
        
        if class_weight is not None and self.label_ignore is not None:
            class_weight[self.label_ignore] = class_weight[-1]
            class_weight_new = class_weight[:-1]
            class_weight = class_weight_new
        self.class_weight = class_weight
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight, size_average=False)
        
        self.losses = collections.defaultdict(list)
        
    def fit(self):
        self.train_loader = dataLoader(meta_data_file=self.train_meta_file, batch_data_dir= self.train_data_dir, id_map=self.id_map, random_shuffle=True)
        if self.valid_data_dir is not None:
            self.valid_loader = dataLoader(meta_data_file=self.valid_meta_file, batch_data_dir= self.valid_data_dir, id_map=self.id_map, random_shuffle=False)

        for epoch in range(self.epoch_start, self.n_epochs + 1):
            train_loss = self._fit_epoch(epoch)
            self.losses["train"].append(train_loss)
            row = 'Epoch: {0:^3}  train: {1:^10.5f}'.format(epoch, self.losses['train'][-1])
            if self.valid_data_dir is not None:
                valid_loss = self._validation_loss()
                self.losses["valid"].append(valid_loss)
                row += 'valid: {0:^10.5f}'.format(self.losses["valid"][-1])
                is_best = valid_loss < self.best_loss
                self.best_loss = min(valid_loss, self.best_loss)
            self._save_checkpoint({
                "epoch": epoch,
                "model": str(self.model),
                "state_dict": self.model.state_dict(),
                "best_loss": self.best_loss,
                "optimizer": self.optimizer.state_dict(),                                      # for changing lr
                "is_best": is_best
            }, is_best)
            print row
            with open(self.log_file, "a") as logf:
                logf.write(row + "\n")
    
    def _fit_epoch(self, epoch=1):
        loss_sum = torch.FloatTensor([0.0])
        n_samp = 0
        #pbar = tqdm(self.train_loader.batch_generate(batch_size=self.batch_size), total=self.train_loader.N_reacs/self.batch_size, desc='({0:^3})'.format(epoch))
        #for batch_idx, sample_batched in pbar:
        for batch_idx, sample_batched in self.train_loader.batch_generate(batch_size=self.batch_size):
#         ## for too small batch size ##
#         for batch_idx, sample_batched in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            uids, iids, lids = map(lambda x: Variable(x), sample_batched)
            preds = self.model(uids, iids)
            loss = self.loss_fn(preds, lids)
            loss_avg = loss/lids.size()[0]
            loss_avg.backward()
            
            self.optimizer.step()
            
            loss_sum += loss.data[0]

            ### test ###
            try:
                assert not np.isnan(loss.data[0])
            except AssertionError as e:
                with open("error_log", "a") as logf:
                    for param in self.model.parameters():
                        logf.write(str(param.data))
            #pbar.set_postfix(train_loss=loss.data[0]/lids.size()[0])
            n_samp += lids.size()[0]
        loss_sum /= n_samp
        return loss_sum[0]
    
    def _validation_loss(self):
        loss_sum = torch.FloatTensor([0.0])
        n_samp = 0
        for batch_idx, sample_batched in self.valid_loader.batch_generate(batch_size=self.batch_size, keep_complete=False):
            uids, iids, lids = map(lambda x: Variable(x), sample_batched)
            preds = self.model(uids, iids)
            loss = self.loss_fn(preds, lids)
            
            loss_sum += loss.data[0]
            n_samp += lids.size()[0]
        loss_sum /= n_samp
        return loss_sum[0]
    
    def _save_checkpoint(self, state, is_best, filename = None):
        if filename is None:
            filename = self.checkpoint_file
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, self.bestpoint_file)
            
    def performance(self, test_data_dir, test_meta_file, ec_sets=None):
        results = []
        for id_map_filtered in id_map_filter_ec_sets(id_map=self.id_map, ec_sets=ec_sets):
            self.test_loader = dataLoader(meta_data_file=test_meta_file, batch_data_dir= test_data_dir, id_map=id_map_filtered, random_shuffle=False)
            if self.test_loader.N_reacs == 0:
                result = [None, None, None, None]
            for _, sample_batched in self.test_loader.batch_generate(batch_size=max(1, self.test_loader.N_reacs)):
                uids, iids, lids = map(lambda x: Variable(x), sample_batched)
                preds_raw = self.model(uids, iids)
                softmax = torch.nn.Softmax()
                preds = softmax(preds_raw).data.numpy()
                n_samp, n_labels = preds.shape
                lids_true = lids.data.numpy()
                if n_samp == 0:
                    result = [None, None, None, None]
                result = evaluate(preds, lids_true)    
            results.append(result)
        return results


def id_map_filter_ec_sets(id_map, ec_sets):
    """
    :param id_map: from post_id to doc_id
    :param ec_sets:
    :return:
    """
    if ec_sets is None:
        return [id_map]
    current_ec_set = set()
    id_maps = []
    for i in range(len(ec_sets)):
        ec_set = set([ec[0] for ec in ec_sets[i]])
        current_ec_set = current_ec_set.union(ec_set)
        id_map_new = dict()
        for id in id_map:
            if id_map[id] in current_ec_set:
                id_map_new[id] = id_map[id]
        id_maps.append(id_map_new)
    return id_maps


if __name__ == "__main__":
    
    data_dir = "/storage/home/jpz5181/work/PSEM/data/CNN_foxnews/"
    data_prefix = "_CNN_foxnews_combined_K10"
    id_map_file = data_dir + "id_map" + data_prefix
    id_map, id_map_reverse = cPickle.load(open(id_map_file, "r"))   
    """
    batch_rBp_dir = data_dir + "K10_batch_train/"
    batch_valid_on_shell_dir = data_dir + "K10_batch_on_shell/valid/"
    batch_valid_off_shell_dir = data_dir + "K10_batch_off_shell/valid/"
    batch_test_on_shell_dir = data_dir + "K10_batch_on_shell/test/"
    batch_test_off_shell_dir = data_dir + "K10_batch_off_shell/test/"
    meta_data_train_file = data_dir + "meta_data_train" + data_prefix
    meta_data_on_valid_file = data_dir + "meta_data_on_shell_valid" + data_prefix
    meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid" + data_prefix
    meta_data_off_test_file = data_dir + "meta_data_off_shell_test" + data_prefix
    """
    data_name = sys.argv[1]
    data_dir = "/storage/home/jpz5181/work/PSEM/data/" + data_name + "/"

    batch_rBp_dir = data_dir + "train/"
    batch_valid_on_shell_dir = data_dir + "on_shell/valid/"
    batch_valid_off_shell_dir = data_dir + "off_shell/valid/"
    batch_test_on_shell_dir = data_dir + "on_shell/test/"
    batch_test_off_shell_dir = data_dir + "off_shell/test/"

    meta_data_train_file = data_dir + "meta_data_train"
    meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid"
    meta_data_off_test_file = data_dir + "meta_data_off_shell_test"
    meta_data_on_valid_file = data_dir + "meta_data_on_shell_valid"
    meta_data_on_test_file = data_dir + "meta_data_on_shell_test"
    
    log_dir = "logs/" + data_name + "/"
    ckpt_dir = "ckpt/" + data_name + "/"
    
    with open(meta_data_train_file, "r") as meta_data_f:
        meta_data = cPickle.load(meta_data_f)
        N_users = meta_data["U"]
        N_items = meta_data["D"]
        N_labels = meta_data["E"]
        N_reacs = sum(meta_data["Md"])
        print "N_u, N_i, N_e, N_r", N_users, N_items, N_labels, N_reacs


    Model = eval(sys.argv[2])
    batch_size = int(sys.argv[3])
    hyperparameters = []
    for i in range(4, len(sys.argv)):
        hyperparameters.append(int(sys.argv[i]))

    exp = Experiment(train_data_dir=batch_rBp_dir, train_meta_file=meta_data_train_file, id_map=id_map_reverse,
                     valid_data_dir=batch_valid_on_shell_dir, valid_meta_file=meta_data_on_valid_file,
                     n_users=N_users, n_items=N_items, n_labels=N_labels, label_ignore=None,
                     method=Model, hyperparameters=hyperparameters, lr=0.01, batch_size=batch_size, class_weight = None,
                     log_file=log_dir, checkpoint_dir=ckpt_dir,
                     resume=ckpt_dir + str(Model.__name__) + "_" + "H".join(map(str,hyperparameters)) + "_" + "best.tar")
#                     resume=None)
#    exp.fit()
    ec_sets = cPickle.load(open("/storage/home/jpz5181/work/PSEM/result/%s_train_posts_divide" % data_name, 'rb'))
    result = exp.performance(test_data_dir=batch_test_on_shell_dir, test_meta_file=meta_data_on_test_file, ec_sets=ec_sets)
    with open("result/" + exp.log_file[5:], "a") as rf:
        rf.write(str(result)+"\n")
