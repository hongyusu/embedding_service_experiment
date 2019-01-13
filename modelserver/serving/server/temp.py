import numpy as np
import zmq

import tensorflow as tf

class Estimator():
    def predict(self, input_fn_builder, yield_single_examples):
        x = input_fn_builder()
        for r in x():
            yield {'client_id':r['client_id'], 'encodes':np.array([[0]])}
            #return [{'client_id':r['client_id'], 'encodes':np.array([[0]])}]


class EstimatorSpec():
    def predict(self):
        pass
