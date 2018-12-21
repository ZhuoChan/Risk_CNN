import os
import tensorflow as tf
from datetime import datetime
import time
import importlib
import numpy as np
from sklearn.utils import shuffle
import tflearn
from data_process import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 保存参数
def save_variables_and_metagraph(sess, saver, model_dir, model_name, global_step):
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)

class Trainer:
    def __init__(self, config, seq_len=None, isTest=False):
        #初始化超参数
        self.init_para(config, isTest)

        if seq_len == None:
            self.seq_len = config.seq_len
        else:
            self.seq_len = seq_len

        self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len], name='inputs')
        inputs_3 = tf.expand_dims(self.inputs_, 2)  # 对样本进行升维，[None,seq_len,1]
        self.labels_ = tf.placeholder(tf.float32, [None, self.n_classes], name='labels')
        self.keep_prob_ = tf.placeholder(tf.float32, name='keep')
        self.learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
        self.global_step = tf.Variable(0, trainable=False)

        # 参数优化
        network = importlib.import_module(self.model_def)
        self.logits = network.inference(inputs_3, self.keep_prob_, self.n_classes)
        self.result = tf.nn.softmax(self.logits)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_))
        #self.cost=mycost( tf.argmax(self.logits, 1), tf.argmax(self.labels_, 1))

        if self.optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(self.learning_rate_)
        elif self.optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(self.learning_rate_, rho=0.9, epsilon=1e-6)
        elif self.optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(self.learning_rate_)
        elif self.optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(self.learning_rate_, decay=0.9, momentum=0.9, epsilon=1.0)
        elif self.optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(self.learning_rate_, 0.9, use_nesterov=True)

        self.optimizer = opt.minimize(self.cost, global_step=self.global_step)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    # 初始化超参数
    def init_para(self, config, isTest):
        # 超参数
        self.n_classes = config.n_classes
        self.model_def = config.model_def
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.subdir = config.name
        self.weight_decay = config.weight_decay

        if isTest == False:
            self.batch_size = config.batch_size  # Batch size
            # self.seq_len = config.seq_len  # Number of steps
            self.epochs = config.epochs
            self.keep_probability = config.keep_probability
            if config.name == None:
                self.subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
            self.log_dir = os.path.join(os.path.expanduser(config.logs_base_dir), self.subdir)
            if not os.path.isdir(self.log_dir):  # Create the log directory if it doesn't exist
                os.makedirs(self.log_dir)
            self.model_dir = os.path.join(os.path.expanduser(config.models_base_dir), self.subdir)
            if not os.path.isdir(self.model_dir):  # Create the model directory if it doesn't exist
                os.makedirs(self.model_dir)

            print('Model directory: %s' % self.model_dir)
            print('Log directory: %s' % self.log_dir)

    def run_epoch(self, X_tr, y_tr, X_vld, y_vld, X_test, y_test):
        self.validation_acc = []
        self.validation_loss = []

        self.train_acc = []
        self.train_loss = []

        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        #min_los = float("inf")
        max_acc = 0
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            iteration = 1

            # Loop over epochs
            for e in range(self.epochs):

                # Loop over batches
                for x, y in self.get_batches(X_tr, y_tr, self.batch_size):

                    # Feed dictionary
                    feed = {self.inputs_: x, self.labels_: y, self.keep_prob_: self.keep_probability,
                            self.learning_rate_: self.learning_rate}

                    # Loss
                    loss, _, acc, step = sess.run([self.cost, self.optimizer, self.accuracy, self.global_step], feed_dict=feed)
                    self.train_acc.append(acc)
                    self.train_loss.append(loss)

                    # Print at each 5 iters
                    if (iteration % 5 == 0):
                        print("Epoch: {}/{}".format(e, self.epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:.6f}".format(acc))

                        #记录
                        summary = tf.Summary()
                        summary.value.add(tag="train_acc", simple_value=acc)
                        summary.value.add(tag='train_loss', simple_value=loss)
                        summary_writer.add_summary(summary, global_step=step)

                    # Compute validation loss at every 10 iterations
                    if (iteration % 1000 == 0):
                        validation_acc = []
                        validation_loss = []
                        res_vec = []
                        for x_val, y_val in self.get_batches(X_vld, y_vld, self.batch_size*5):
                            feed = {self.inputs_: x_val, self.labels_: y_val, self.keep_prob_: 1.0}
                            loss_v, acc_v, step, res = sess.run([self.cost, self.accuracy, self.global_step, self.result], feed_dict=feed)
                            validation_acc.append(acc_v)
                            validation_loss.append(loss_v)
                            res_vec.append(res[:, 1])
                        # 计算最后一点数据
                        rest = len(X_vld) % (self.batch_size * 50)
                        x_val = X_vld[- rest:, :]
                        y_val = y_vld[- rest:, :]
                        feed = {self.inputs_: x_val, self.labels_: y_val, self.keep_prob_: 1.0}
                        loss_v, acc_v, step, res = sess.run([self.cost, self.accuracy, self.global_step, self.result], feed_dict=feed)
                        validation_acc.append(acc_v)
                        validation_loss.append(loss_v)
                        res_vec.append(res[:, 1])

                        mean_acc = np.mean(validation_acc)
                        mean_loss = np.mean(validation_loss)

                        # Print info
                        print("Epoch: {}/{}".format(e, self.epochs),
                              "Iteration: {:d}".format(iteration),
                              "Validation loss: {:6f}".format(mean_loss),
                              "Validation acc: {:.6f}".format(mean_acc))

                        summary = tf.Summary()
                        summary.value.add(tag="val_acc", simple_value=mean_acc)
                        summary.value.add(tag="val_error", simple_value= (1 - mean_acc))
                        summary.value.add(tag='val_loss', simple_value=mean_loss)
                        summary_writer.add_summary(summary, global_step=step)

                        # Store
                        self.validation_acc.append(acc_v)
                        self.validation_loss.append(loss_v)

                        # 计算测试集相关数据
                        predict = []
                        for x, y in self.get_batches(X_test, y_test, self.batch_size * 50):
                            feed = {self.inputs_: x, self.labels_: y, self.keep_prob_: 1.0}
                            pre = sess.run(tf.argmax(self.result, 1), feed_dict=feed)
                            predict += list(pre)
                        # 计算测试集最后一点数据
                        rest = len(X_test) % (self.batch_size * 50)
                        x = X_test[- rest:, :]
                        y = y_test[- rest:, :]
                        feed = {self.inputs_: x, self.labels_: y, self.keep_prob_: 1.0}
                        pre = sess.run(tf.argmax(self.result, 1), feed_dict=feed)
                        predict += list(pre)
                        predict = np.array(predict)
                        accuracy = accuracy_score(y_test[:,1], predict)
                        precision = precision_score(y_test[:,1], predict)
                        recall = recall_score(y_test[:,1], predict)
                        f1 = f1_score(y_test[:,1], predict)

                        summary = tf.Summary()
                        summary.value.add(tag="test_acc", simple_value=accuracy)
                        summary.value.add(tag="test_error", simple_value=(1-accuracy))
                        summary.value.add(tag="test_recall", simple_value=recall)
                        summary.value.add(tag='test_precision', simple_value=precision)
                        summary.value.add(tag='test_f1', simple_value=f1)
                        summary_writer.add_summary(summary, global_step=step)

                        if max_acc < mean_acc:
                            max_acc = mean_acc
                            save_variables_and_metagraph(sess, saver, self.model_dir, self.subdir, step)
                    # Iterate
                    iteration += 1

    def prediction(self, X, model_dir=None):
        sess = tf.Session()
        with sess.as_default():
            # Restore
            saver = tf.train.Saver()
            if model_dir == None:
                saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
            else:
                saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            feed = {self.inputs_: X, self.keep_prob_: 1.0}
            #log= sess.run(self.result, feed_dict=feed)
            log= sess.run(tf.argmax(self.logits, 1), feed_dict=feed)
            return log

    #取batch_size个样本的生成器
    def get_batches(self,X, y, batch_size=100):
        # 每次取batch_size个样本
        n_batches = len(X) // batch_size
        X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]
        # 生成器
        for b in range(0, len(X), batch_size):
            yield X[b:b + batch_size], y[b:b + batch_size]
