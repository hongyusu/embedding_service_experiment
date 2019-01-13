#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
from multiprocessing import Process
from collections import defaultdict
import zmq
import zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi
import numpy as np

from .helper import *
from .Command import Command

class Sink(Process):
    def __init__(self, args, front_sink_addr):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'), args.verbose)
        self.front_sink_addr = front_sink_addr

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def _run(self, receiver, frontend, sender):
        receiver_addr = auto_bind(receiver)
        frontend.connect(self.front_sink_addr)
        sender.bind('tcp://*:%d' % self.port)

        pending_checksum = defaultdict(int)
        pending_result = defaultdict(list)
        job_checksum = defaultdict(int)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend
        frontend.send(receiver_addr.encode('ascii'))

        self.logger.info('ready')

        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            if socks.get(receiver) == zmq.POLLIN:
                msg = receiver.recv_multipart()
                job_id = msg[0]
                # parsing the ndarray
                arr_info, arr_val = jsonapi.loads(msg[1]), msg[2]
                X = np.frombuffer(memoryview(arr_val), dtype=arr_info['dtype'])
                X = X.reshape(arr_info['shape'])
                job_info = job_id.split(b'@')
                job_id = job_info[0]
                partial_id = job_info[1] if len(job_info) == 2 else 0
                pending_result[job_id].append((X, partial_id))
                pending_checksum[job_id] += X.shape[0]
                self.logger.info('collect job %s (%d/%d)' % (job_id,
                                                             pending_checksum[job_id],
                                                             job_checksum[job_id]))

                # check if there are finished jobs, send it back to workers
                finished = [(k, v) for k, v in pending_result.items() if pending_checksum[k] == job_checksum[k]]
                for job_info, tmp in finished:
                    self.logger.info('send back\tsize: %d\tjob id:%s\t' % (job_checksum[job_info], job_info))
                    # re-sort to the original order
                    tmp = [x[0] for x in sorted(tmp, key=lambda x: int(x[1]))]
                    client_addr, req_id = job_info.split(b'#')
                    send_ndarray(sender, client_addr, np.concatenate(tmp, axis=0), req_id)
                    pending_result.pop(job_info)
                    pending_checksum.pop(job_info)
                    job_checksum.pop(job_info)

            if socks.get(frontend) == zmq.POLLIN:
                client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()
                if msg_type == Command.new_job:
                    job_info = client_addr + b'#' + req_id
                    job_checksum[job_info] = int(msg_info)
                    self.logger.info('job register\tsize: %d\tjob id: %s' % (int(msg_info), job_info))
                elif msg_type == Command.show_config:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    self.logger.info('send config\tclient %s' % client_addr)
                    sender.send_multipart([client_addr, msg_info, req_id])


