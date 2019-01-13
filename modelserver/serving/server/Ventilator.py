#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import threading
import zmq
import zmq.decorators as zmqd
from termcolor import colored
from datetime import datetime
from multiprocessing.pool import Pool

from .Decorator import multi_socket
from .helper import *
from .Worker import Worker
from .Sink import Sink
from .Statistic import Statistic
from .Command import Command

_tf_ver_ = check_tf_version()
__version__ = '1.0.0'

class Ventilator(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)

        self.model_dir = args.model_dir
        self.max_seq_len = args.max_seq_len
        self.num_worker = args.num_worker
        self.max_batch_size = args.max_batch_size
        self.num_concurrent_socket = max(8, args.num_worker * 2)  # optimize concurrency for multi-clients
        self.port = args.port
        self.args = args
        self.status_args = {k: (v if k != 'pooling_strategy' else v.value) for k, v in sorted(vars(args).items())}
        self.status_static = {
            'tensorflow_version': _tf_ver_,
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }
        self.processes = []
        self.logger.info('freeze, optimize and export graph, could take a while...')
        with Pool(processes=1) as pool:
            # optimize the graph, must be done in another process
            from .graph import optimize_graph
            self.graph_path = pool.apply(optimize_graph, (self.args,))
        # from .graph import optimize_graph
        # self.graph_path = optimize_graph(self.args, self.logger)
        if self.graph_path:
            self.logger.info('optimized graph is stored at: %s' % self.graph_path)
        else:
            raise FileNotFoundError('graph optimization fails and returns empty result')

    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        for p in self.processes:
            p.close()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _send_close_signal(self, _, frontend):
        frontend.connect('tcp://localhost:%d' % self.port)
        frontend.send_multipart([b'', Command.terminate, b'', b''])

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @multi_socket(zmq.PUSH, num_socket='num_concurrent_socket')
    def _run(self, _, frontend, sink, *backend_socks):

        def push_new_job(_job_id, _json_msg, _msg_len):
            # backend_socks[0] is always at the highest priority
            _sock = backend_socks[0] if _msg_len <= self.args.priority_batch_size else rand_backend_socket
            _sock.send_multipart([_job_id, _json_msg])

        # bind all sockets
        self.logger.info('bind all sockets')
        frontend.bind('tcp://*:%d' % self.port)
        addr_front2sink = auto_bind(sink)
        addr_backend_list = [auto_bind(b) for b in backend_socks]
        self.logger.info('open %d ventilator-worker sockets' % len(addr_backend_list))

        # start the sink process
        self.logger.info('start the sink')
        proc_sink = Sink(self.args, addr_front2sink)
        self.processes.append(proc_sink)
        proc_sink.start()
        addr_sink = sink.recv().decode('ascii')

        # start the backend processes
        device_map = self._get_device_map()
        for idx, device_id in enumerate(device_map):
            process = Worker(idx, self.args, addr_backend_list, addr_sink, device_id,
                                 self.graph_path)
            self.processes.append(process)
            process.start()

        rand_backend_socket = None
        server_status = Statistic()
        while True:
            try:
                request = frontend.recv_multipart()
            except ValueError:
                self.logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(request))
                self.logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(request)), exc_info=True)
            else:
                client, msg, req_id, msg_len = request
                server_status.update(request)
                if msg == Command.terminate:
                    break
                elif msg == Command.show_config:
                    self.logger.info('new config request\treq id: %d\tclient: %s' % (int(req_id), client))
                    status_runtime = {'client': client.decode('ascii'),
                                      'num_process': len(self.processes),
                                      'ventilator -> worker': addr_backend_list,
                                      'worker -> sink': addr_sink,
                                      'ventilator <-> sink': addr_front2sink,
                                      'server_current_time': str(datetime.now()),
                                      'statistic': server_status.value,
                                      'device_map': device_map,
                                      'num_concurrent_socket': self.num_concurrent_socket}

                    sink.send_multipart([client, msg, jsonapi.dumps({**status_runtime,
                                                                     **self.status_args,
                                                                     **self.status_static}), req_id])
                else:
                    self.logger.info('new encode request\treq id: %d\tsize: %d\tclient: %s' %
                                     (int(req_id), int(msg_len), client))
                    # register a new job at sink
                    sink.send_multipart([client, Command.new_job, msg_len, req_id])

                    # renew the backend socket to prevent large job queueing up
                    # [0] is reserved for high priority job
                    # last used backennd shouldn't be selected either as it may be queued up already
                    rand_backend_socket = random.choice([b for b in backend_socks[1:] if b != rand_backend_socket])

                    # push a new job, note super large job will be pushed to one socket only,
                    # leaving other sockets free
                    job_id = client + b'#' + req_id
                    if int(msg_len) > self.max_batch_size:
                        seqs = jsonapi.loads(msg)
                        job_gen = ((job_id + b'@%d' % i, seqs[i:(i + self.max_batch_size)]) for i in
                                   range(0, int(msg_len), self.max_batch_size))
                        for partial_job_id, job in job_gen:
                            push_new_job(partial_job_id, jsonapi.dumps(job), len(job))
                    else:
                        push_new_job(job_id, msg, int(msg_len))

        self.logger.info('terminated!')

    def _get_device_map(self):
        self.logger.info('get devices')
        run_on_gpu = False
        device_map = [-1] * self.num_worker
        if not self.args.cpu:
            try:
                import GPUtil
                num_all_gpu = len(GPUtil.getGPUs())
                avail_gpu = GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, self.num_worker))
                num_avail_gpu = len(avail_gpu)

                if num_avail_gpu >= self.num_worker:
                    run_on_gpu = True
                elif 0 < num_avail_gpu < self.num_worker:
                    self.logger.warning('only %d out of %d GPU(s) is available/free, but "-num_worker=%d"' %
                                        (num_avail_gpu, num_all_gpu, self.num_worker))
                    if not self.args.device_map:
                        self.logger.warning('multiple workers will be allocated to one GPU, '
                                            'may not scale well and may raise out-of-memory')
                    else:
                        self.logger.warning('workers will be allocated based on "-device_map=%s", '
                                            'may not scale well and may raise out-of-memory' % self.args.device_map)
                    run_on_gpu = True
                else:
                    self.logger.warning('no GPU available, fall back to CPU')

                if run_on_gpu:
                    device_map = ((self.args.device_map or avail_gpu) * self.num_worker)[: self.num_worker]
            except FileNotFoundError:
                self.logger.warning('nvidia-smi is missing, often means no gpu on this machine. '
                                    'fall back to cpu!')
        self.logger.info('device map: \n\t\t%s' % '\n\t\t'.join(
            'worker %2d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
            enumerate(device_map)))
        return device_map


