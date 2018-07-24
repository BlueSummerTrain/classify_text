#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from model import Model
from sklearn.cross_validation import train_test_split

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_per", 0.01, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("caijing_data_file", "./data/train_data/train_caijing.txt", "Data source for the caijing data.")
tf.flags.DEFINE_string("caipiao_data_file", "./data/train_data/train_caipiao.txt", "Data source for the caipiao data.")
tf.flags.DEFINE_string("fangchan_data_file", "./data/train_data/train_fangchan.txt", "Data source for the fangchan data.")
tf.flags.DEFINE_string("gupiao_data_file", "./data/train_data/train_gupiao.txt", "Data source for the gupiao data.")
tf.flags.DEFINE_string("jiaju_data_file", "./data/train_data/train_jiaju.txt", "Data source for the jiaju data.")
tf.flags.DEFINE_string("jiaoyu_data_file", "./data/train_data/train_jiaoyu.txt", "Data source for the jiaoyu data.")
tf.flags.DEFINE_string("shishang_data_file", "./data/train_data/train_shishang.txt", "Data source for the shishang data.")
tf.flags.DEFINE_string("shizheng_data_file", "./data/train_data/train_shizheng.txt", "Data source for the shizheng data.")
tf.flags.DEFINE_string("tiyu_data_file", "./data/train_data/train_tiyu.txt", "Data source for the tiyu data.")
tf.flags.DEFINE_string("yule_data_file", "./data/train_data/train_yule.txt", "Data source for the yule data.")
tf.flags.DEFINE_string("input_test_file", "./data/test_data/input_test_file.txt", "Data source for the test data.")
tf.flags.DEFINE_string("input_label_file", "./data/test_data/input_label_file.txt", "Label file for test text data source.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 32, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.00005, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("max_sentence_len", 32, "max length of sentences (default: 64)")


# Training parameters
tf.flags.DEFINE_float("learning_rate", 1e-5, "learning rate (default:1e-4)")
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("test_every", 2000, "Test model on test set after this many steps(default:500)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("padding_token", '<pad>', "uniform sentences")

FLAGS = tf.flags.FLAGS
#print FLAGS
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

# Output directory for models and summaries
out_dir = data_helpers.mkdir_if_not_exist("./runs")

# Load data
print("Loading data...")
max_sentence_len = FLAGS.max_sentence_len
x_text, y = data_helpers.load_data_and_labels(FLAGS.caijing_data_file, \
                                              FLAGS.caipiao_data_file, \
                                              FLAGS.fangchan_data_file, \
                                              FLAGS.gupiao_data_file,\
                                              FLAGS.jiaju_data_file,\
                                              FLAGS.jiaoyu_data_file,\
                                              FLAGS.shishang_data_file,\
                                              FLAGS.shizheng_data_file,\
                                              FLAGS.tiyu_data_file,\
                                              FLAGS.yule_data_file)
sentences = data_helpers.padding_sentences(x_text, FLAGS.padding_token,max_sentence_len)

print("len(x_text)",len(x_text))
print("len(y)",len(y))
# Build vocabulary
voc = None
vocsize = None

if os.path.exists('./runs/vocab'):
    # when sess restore,just reload vocab 
    voc,vocsize = data_helpers.read_vocabulary('./runs/vocab') 
else:
    voc,vocsize = data_helpers.build_vocabulary(sentences,'./runs/vocab')

x = np.array(data_helpers.sentence2matrix(sentences,max_sentence_len,voc))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
data_len = len(x_shuffled)
x_train,x_dev,y_train,y_dev= train_test_split(x_shuffled,y_shuffled,test_size=FLAGS.dev_per,random_state=42)
print("Total/Train/Dev: {:d}/{:d}/{:d}".format(data_len,len(y_train), len(y_dev)))

# Training
# ==================================================
global_graph = tf.Graph()

with global_graph.as_default():

    sess = tf.Session(graph=global_graph)

    with sess.as_default():
        cnn = Model(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size = vocsize,
            batch_size = FLAGS.batch_size,
            embedding_size=FLAGS.embedding_dim,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        loss_summary = tf.summary.scalar('loss', cnn.loss)
        acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

        train_summary_op = tf.summary.merge([loss_summary,acc_summary])
        train_summary_dir = os.path.join(out_dir,"summaries","train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir,"summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "model"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)
        checkpoint_file = tf.train.latest_checkpoint('./runs/model')

        if checkpoint_file != None:
            saver.restore(sess, checkpoint_file)
            print ("restore session from checkpoint files")
        else:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())



        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, summaries, loss, l2_loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.l2_losses, cnn.accuracy],feed_dict)
            time_str = datetime.datetime.now().strftime("%H:%M:%S.%f")
            print("train set:*** {}: step {}, loss {:g}, l2_loss {:g}, acc {:g}".format(time_str, step, loss, l2_loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

            return loss

        def dev_step(x_batch, y_batch, writer = None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }

            step, summaries, loss, l2_loss, accuracy = sess.run([global_step,dev_summary_op, cnn.loss, cnn.l2_losses, cnn.accuracy],feed_dict)
            time_str = datetime.datetime.now().strftime("%H:%M:%S.%f")
            print("dev set:***{}: step {}, loss {:g}, l2_loss {:g}, acc {:g}".format(time_str, step, loss, l2_loss, accuracy))

            if writer:
                writer.add_summary(summaries, step)

            return loss

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        min_dev_loss = 1000
        current_patience = 0

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_loss=train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:

                print("\nEvaluation on dev set:")
                dev_loss = dev_step(x_dev, y_dev, dev_summary_writer)

                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    print('current loss : %f'%(min_dev_loss))
                else:
                     current_patience += 1
                     print('no improvement,current_loss/last_loss=%f/%f'%(dev_loss,min_dev_loss))

            if current_step % FLAGS.test_every == 0:
                print("Loading  Test data...")

                x_raw, y_data = data_helpers.load_testfile_and_labels(FLAGS.input_test_file, FLAGS.input_label_file,4)
                sentences= data_helpers.padding_sentences(x_raw, FLAGS.padding_token, FLAGS.max_sentence_len)
                x_test = np.array(data_helpers.sentence2matrix(sentences, FLAGS.max_sentence_len, voc))
                y_test = []

                for item in y_data:
                    if item == 0:
                        y_test.append(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
                    elif item == 1:
                        y_test.append(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
                    elif item == 2:
                        y_test.append(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0 ,0]))
                    elif item == 3:
                        y_test.append(np.array([0, 0, 0, 1, 0, 0 ,0, 0, 0, 0]))
                    elif item == 4:
                        y_test.append(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
                    elif item == 5:
                        y_test.append(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
                    elif item == 6:
                        y_test.append(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
                    elif item == 7:
                        y_test.append(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
                    elif item == 8:
                        y_test.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
                    else:
                        y_test.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))

                y_test = tuple(y_test)
                step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy],{cnn.input_x:x_test[:128], cnn.input_y:y_test[:128], cnn.dropout_keep_prob: 1.0})
                time_str = datetime.datetime.now().isoformat()
                log_str = "Time::{}, Step::{}, Loss::{}, Accuracy::{} on Test data.".format(time_str, current_step, loss, accuracy)
                print(log_str)

                with open(out_dir+'/training_log.txt', 'a') as out_put_file:
                    out_put_file.write(log_str + '\n')
                tf.train.write_graph(sess.graph_def, checkpoint_dir, 'classify_text.pbtxt')
                saver.save(sess, checkpoint_dir + '/classify_text.ckpt', global_step=step)
                print("Accuray ::{},Save model checkpoint to {}\n".format(accuracy, checkpoint_dir))
