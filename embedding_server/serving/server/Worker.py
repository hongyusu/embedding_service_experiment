#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import multiprocessing
from multiprocessing import Process
import zmq
import zmq.decorators as zmqd
from termcolor import colored

from .Decorator import multi_socket
from zmq.utils import jsonapi
from .helper import *

class Worker(Process):
    def __init__(self, id, args, worker_address_list, sink_address, device_id, graph_path):
        super().__init__()
        self.worker_id = id
        self.device_id = device_id
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), args.verbose)
        self.max_seq_len = args.max_seq_len
        self.mask_cls_sep = args.mask_cls_sep
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.model_dir = args.model_dir
        self.verbose = args.verbose
        self.graph_path = graph_path

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def get_estimator(self, tf):
        from .Estimator import Estimator
        from .Estimator import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0'])

            return EstimatorSpec(mode=mode, predictions={
                'client_id': features['client_id'],
                'encodes': output[0]
            })

        return Estimator()

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, sink, *receivers):
        self.logger.info('use device %s, load graph from %s' %
                         ('cpu' if self.device_id < 0 else ('gpu: %d' % self.device_id), self.graph_path))

        tf = import_tf(self.device_id, self.verbose)
        estimator = self.get_estimator(tf)

        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)

        sink.connect(self.sink_address)
        for r in estimator.predict(self.input_fn_builder(receivers, tf), yield_single_examples=False):
            send_ndarray(sink, r['client_id'], r['encodes'])
            self.logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))

    def input_fn_builder(self, socks, tf):
        from .bert.extract_features import convert_lst_to_features
        from .bert.tokenization import FullTokenizer

        def gen():
            tokenizer = FullTokenizer(vocab_file=os.path.join(self.model_dir, 'vocab.txt'))
            poller = zmq.Poller()
            for sock in socks:
                poller.register(sock, zmq.POLLIN)

            self.logger.info('ready and listening!')

            while not self.exit_flag.is_set():
                events = dict(poller.poll())
                for sock_idx, sock in enumerate(socks):
                    if sock in events:
                        client_id, raw_msg = sock.recv_multipart()
                        msg = jsonapi.loads(raw_msg)
                        self.logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg), client_id))
                        # check if msg is a list of list, if yes consider the input is already tokenized
                        is_tokenized = all(isinstance(el, list) for el in msg)
                        tmp_f = list(convert_lst_to_features(msg, self.max_seq_len, tokenizer, self.logger,
                                                             is_tokenized, self.mask_cls_sep))
                        yield {
                            'client_id': client_id,
                            'input_ids': [f.input_ids for f in tmp_f],
                            'input_mask': [f.input_mask for f in tmp_f],
                            'input_type_ids': [f.input_type_ids for f in tmp_f]
                        }

        def input_fn():
            return gen
#            return (tf.data.Dataset.from_generator(
#                gen,
#                output_types={'input_ids': tf.int32,
#                              'input_mask': tf.int32,
#                              'input_type_ids': tf.int32,
#                              'client_id': tf.string},
#                output_shapes={
#                    'client_id': (),
#                    'input_ids': (None, self.max_seq_len),
#                    'input_mask': (None, self.max_seq_len),
#                    'input_type_ids': (None, self.max_seq_len)}).prefetch(self.prefetch_size))

        return input_fn


