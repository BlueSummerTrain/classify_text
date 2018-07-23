import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class Model(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by 3*( convolutional, max-pooling) and 3*FC and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size, batch_size, embedding_size, l2_reg_lambda):

        print "Model __init__"
        self.batch_size = batch_size
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_data")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_label")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-0.1,0.1),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W,self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,-1)


        # Creat  3*( convolution + maxpool) layer
        with slim.arg_scope([slim.conv2d], padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.05), data_format='NHWC'):
            net_1 = slim.repeat(self.embedded_chars_expanded, 1,slim.conv2d, 128, [5, 5], scope='conv1')
            net_1 = slim.max_pool2d(net_1, [2,2], scope='pool1')
            net_1 = tf.nn.dropout(net_1, self.dropout_keep_prob)

            net_2 = slim.repeat(net_1, 1, slim.conv2d, 128, [3,3], scope='conv2')
            net_2 = slim.max_pool2d(net_2, [2, 2], scope='pool2')
            net_2 = tf.nn.dropout(net_2, self.dropout_keep_prob)

            net_3 = slim.repeat(net_2, 1, slim.conv2d, 64, [3,3], scope='conv3')
            net_3 = slim.max_pool2d(net_3, [2, 2], scope='pool3')
            net_3 = tf.nn.dropout(net_3, self.dropout_keep_prob)


            # Creat 3*FC layer
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.05)):
            #self.fc_input_x = tf.reshape(net_3, [-1, 4*32*16])
            self.fc_input_x = tf.reshape(net_3, [-1, net_3.shape[1] * net_3.shape[2] * net_3.shape[3]])

            fc_net = slim.fully_connected(self.fc_input_x, 1024, scope='fc1')
            fc_net = slim.dropout(fc_net,self.dropout_keep_prob,scope='fc_drop1')
            fc_net = slim.fully_connected(fc_net, 1024, scope='fc2')

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):

            W = tf.get_variable(
                "W",
                shape=[1024, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            self.scores = tf.nn.xw_plus_b(fc_net, W, b, name="scores")
            self.softmax_data = tf.nn.softmax(self.scores,name="cf_softmax")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss and l2_loss
        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                         labels=self.input_y,name="cross_entropy")
            vars = tf.trainable_variables()
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2_reg_lambda

            self.loss = tf.reduce_mean(self.cross_entropy + self.l2_losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            #error_predictions = tf.equal(self.predictions, tf.arg_min(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



#使用gpu运行代码步骤
#先找到需要上传的文件上传本地文件到服务器
#scp -r 文件夹/tclxa@10.120.105.206:/home/tcl/workspace/liujun/

#进入服务器
#ssh tclxa@10.120.105.206
#xa123#321
#进入workspace目录查看gpu使用情况
#nvidia-smi
#然后进入到运行文件的上一级根目录
#CUSA_VISIBLE_DEVICES=？ python train.py
