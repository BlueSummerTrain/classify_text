#! /usr/bin/env python
#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from model import Model
import csv

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# Parameters
# ==================================================

# Data Parameters
#tf.flags.DEFINE_string("input_test_file", "./data/test_data/input_test_file.txt", "Data source for the positive data.")
#tf.flags.DEFINE_string("input_label_file", "./data/test_data/input_label_file.txt", "Label file for test text data source.")


tf.flags.DEFINE_string("input_test_file", "./data/test_self.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("input_label_file", "", "Label file for test text data source.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/model", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
tf.flags.DEFINE_integer("max_document_length", 32, "max document length (default: 64)")
tf.flags.DEFINE_integer("num_labels", 5, "num labels (default: 2)")
tf.flags.DEFINE_string("padding_token", '<pad>', "uniform sentences")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test=data_helpers.load_testfile_and_labels(FLAGS.input_test_file, FLAGS.input_label_file, FLAGS.num_labels)

# Get Embedding vector x_test
sentences= data_helpers.padding_sentences(x_raw, FLAGS.padding_token,FLAGS.max_document_length)
print "sentences length : %d" % len(sentences)
voc,_ = data_helpers.read_vocabulary('./runs/vocab')
print "voc length : %d" % len(voc)

x_test = np.array(data_helpers.sentence2matrix(sentences,FLAGS.max_document_length,voc))
print("x_test.shape = {}".format(x_test.shape))
print("\nEvaluating...\n")


# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
#checkpoint_file = "./runs/model/classify_text.ckpt-1000"
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        input_x = graph.get_operation_by_name("input_data").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:

    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    erorr_predictions = []
    error_x_raw = []

    for pre_index in range(len(all_predictions)):
        if all_predictions[pre_index] != y_test[pre_index]:
            result = "LABEL::"+ str(y_test[pre_index])+" ,ERROR_PRE:: " + str(all_predictions[pre_index])
            erorr_predictions.append(result)
            error_x_raw.append(x_raw[pre_index])

    errors_human_readable = np.column_stack((np.array(error_x_raw), erorr_predictions))
    error_path = os.path.join(FLAGS.checkpoint_dir, "..", "erorr_prediction.txt")
    print("Saving Error evaluation to {0}".format(error_path))

    with open(error_path,'w') as f:
        csv.writer(f).writerows(errors_human_readable)

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.txt")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
