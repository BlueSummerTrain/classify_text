#!/usr/bin/env python
# encoding=utf-8
import tensorflow as tf
import numpy as np
import data_helpers
from tensorflow.python.tools import freeze_graph

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def create_pbfile():
    saved_graph_name = './runs/model/classify_text.pbtxt'
    saved_ckpt_name = './runs/model/classify_text.ckpt-30000'
    output_frozen_graph_name = './runs/cls_name.pb'

    freeze_graph.freeze_graph(input_graph=saved_graph_name, input_saver='', \
                              input_binary=False, \
                              input_checkpoint=saved_ckpt_name,\
                              output_node_names='output/cf_softmax', \
                              restore_op_name='', \
                              filename_tensor_name='', \
                              output_graph=output_frozen_graph_name,\
                              clear_devices=True, \
                              initializer_nodes='')

def get_cf_batch_data(str_data,vocab_path):
    x_data = data_helpers.read_data_from_str(str_data,32)
    voc,_ = data_helpers.read_vocabulary(vocab_path)
    x_test = np.array(data_helpers.sentence2matrix(x_data,32,voc))
    batches = data_helpers.batch_iter(list(x_test), 128, 1, shuffle=False)
    return batches

def test_freeze_model():
    FROZEN_MODEL = './runs/cls_name.pb'
    INPUT_NODE_1 = 'input_data:0'
    INPUT_NODE_2 = 'dropout_keep_prob:0'
    OUTPUT_NODE_1 = 'output/cf_softmax:0'

    with tf.gfile.GFile(FROZEN_MODEL,"rb") as f:
        graph_o = tf.GraphDef()
        graph_o.ParseFromString(f.read())

    with tf.Graph().as_default() as G:
        tf.import_graph_def(graph_o,\
                            input_map=None,\
                            return_elements=None,\
                            name='',\
                            op_dict=None,\
                            producer_op_list=None)

    x1 = G.get_tensor_by_name(INPUT_NODE_1)
    x2 = G.get_tensor_by_name(INPUT_NODE_2)
    y = G.get_tensor_by_name(OUTPUT_NODE_1)

    with  tf.Session(graph=G) as sess:

        get_batches = get_cf_batch_data("捉妖记",'./runs/vocab')

        for x_test_batch in get_batches:
            batch_predictions = sess.run(y, {x1: x_test_batch, x2: 1.0})
            predict_data = batch_predictions[0].tolist()
            print predict_data.index(max(predict_data))

if __name__ == '__main__':
        create_pbfile()
        print 'save to pb file ok...............................'
        test_freeze_model() 
        print 'test freeze file ok.................'
