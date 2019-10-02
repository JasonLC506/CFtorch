import numpy as np
# from functions import multinomial, probNormalize
import cPickle
from multiprocessing import Process, Queue
import os
from datetime import datetime
import random
import torch

class dataLoader(object):
    def __init__(self, meta_data_file, batch_data_dir, id_map, max_qsize = 10000000, random_shuffle=False):

        self.E = 0                                  # dimension of emotion
        self.U = 0                                  # number of users
        self.N_reacs = 0                            # number of total reactions in this data

        self.batch_data_dir = batch_data_dir
        self.id_map = id_map
        with open(meta_data_file, "r") as f:
            meta_data = cPickle.load(f)
            self.E = meta_data["E"]
            self.U = meta_data["U"]
            self.N_reacs = sum([meta_data["Md"][self.id_map[post_id]] for post_id in self.id_map])   # count valid N_reacs


        self.random_shuffle = random_shuffle

        # multiprocess data reader #
        self.data_queue = Queue(maxsize=max_qsize)          # data queue for multiprocess
        self.data_reader = Process(target=self._dataBatchReader, args=(self.data_queue,))
        self.data_reader.daemon = True              # daemon process, killed automatically when main process ended

        self.data_reader.start()                    # !!! start when instantiated !!!

    def _dataBatchReader(self, data_queue, timeout=10000):
        while True:
            file_list = os.listdir(self.batch_data_dir)

            if self.random_shuffle:
                random.shuffle(file_list)
            for fn in file_list:
                start = datetime.now()
                with open(os.path.join(self.batch_data_dir, fn), "r") as f:
                    posts = cPickle.load(f)
                duration = datetime.now() - start
                # print "_dataBatchReader: load %s takes %f s" % (fn, duration.total_seconds())
                post_id_list = posts.keys()
                if self.random_shuffle:
                    random.shuffle(post_id_list)
                for post_id in post_id_list:
                    if post_id not in self.id_map:
                        continue
                    doc_id = self.id_map[post_id]
                    doc_u, doc_e = posts[post_id]
                    for n in range(len(doc_u)):
                        data_queue.put([doc_u[n], doc_id, doc_e[n]], block=True, timeout=timeout)
                        
    def batch_generate(self, batch_size, keep_complete=True):
        """ always complete batch """
        N_batch = int(self.N_reacs) / int(batch_size)
        residue = int(self.N_reacs) % int(batch_size)
        if residue != 0:
            N_batch += 1 
        for i_batch in xrange(N_batch):
            if not keep_complete and residue != 0 and i_batch == (N_batch-1):
                yield i_batch, self.generate_single_batch(residue)
            else:
                yield i_batch, self.generate_single_batch(batch_size)
    
    def generate_single_batch(self, batch_size):
        doc_us = []
        doc_ids = []
        doc_es = []
        for i_samp in xrange(batch_size):
            [doc_u, doc_id, doc_e] = self.data_queue.get(block=True, timeout=1000)
            doc_us.append(doc_u)
            doc_ids.append(doc_id)
            doc_es.append(doc_e)
        data_batched =[doc_us, doc_ids, doc_es]
        return map(lambda x: torch.LongTensor(x), data_batched)


if __name__ == "__main__":
    """
    data_dir = "/storage/home/jpz5181/work/LDA_CF/data/CNN_foxnews/"
    data_prefix = "_CNN_foxnews_combined_K10"
    batch_rBp_dir = data_dir + "K10_batch_train/"
    batch_valid_on_shell_dir = data_dir + "K10_batch_on_shell/valid/"
    batch_valid_off_shell_dir = data_dir + "K10_batch_off_shell/valid/"
    batch_test_on_shell_dir = data_dir + "K10_batch_on_shell/test/"
    batch_test_off_shell_dir = data_dir + "K10_batch_off_shell/test/"
    meta_data_train_file = data_dir + "meta_data_train" + data_prefix
    meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid" + data_prefix
    meta_data_off_test_file = data_dir + "meta_data_off_shell_test" + data_prefix
    id_map_file = data_dir + "id_map" + data_prefix
    id_map, id_map_reverse = cPickle.load(open(id_map_file, "r"))
    
    data_train = dataLoader(meta_data_file=meta_data_train_file, batch_data_dir= batch_rBp_dir, id_map=id_map_reverse, random_shuffle=True)
    print "E, U, N_reacs", data_train.E, data_train.U, data_train.N_reacs
    for i_batch, data_batched in data_train.batch_generate(10):
        print i_batch
        print data_batched
        print type(data_batched[0])
        break 
    """
    data_dir = "/storage/home/jpz5181/work/LDA_CF/data/CNN_foxnews/"
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
    data_dir = "/storage/home/jpz5181/work/LDA_CF/data/period_foxnews_nolike/"

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

    data_train = dataLoader(meta_data_file=meta_data_train_file, batch_data_dir=batch_rBp_dir, id_map=id_map_reverse, random_shuffle=True)
    print "E, U, N_reacs", data_train.E, data_train.U, data_train.N_reacs
    for i_batch, data_batched in data_train.batch_generate(10):
        print i_batch
        print data_batched
        print type(data_batched[0])
        break
    data_valid = dataLoader(meta_data_file=meta_data_on_valid_file, batch_data_dir=batch_valid_on_shell_dir, id_map=id_map_reverse, random_shuffle=False)
    print "E, U, N_reacs", data_valid.E, data_valid.U, data_valid.N_reacs
    for i_batch, data_batched in data_valid.batch_generate(10):
        print i_batch
        print data_batched
        print type(data_batched[0])
        break
