from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_process import *
from trainer import *
from trainer import Trainer
import argparse
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='./logs')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='./my_models')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.CNN')
    parser.add_argument('--train_data_file', type=str,
                        help='Path to the train data.',
                        default='../atec_anti_fraud_train_0.25.csv')
    parser.add_argument('--seq_len', type=str,
                        help='Number of seq length.',
                        default=44)
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to run.', default=50)
    parser.add_argument('--batch_size', type=int,
                        help='Number of data to process in a batch.', default=256)
    parser.add_argument('--n_classes', type=str,
                        help='Number of class.', default=2)
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=0.5)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=5e-4)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.00001)
    parser.add_argument('--name', type=str,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=None)
    return parser.parse_args(argv)

def main_atec(config):
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allocator_type ='BFC'
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.90

    #X_tr, label_tr, X_vld, label_vld,X_test,label_test = read_data()  # 读取数据
    X_tr, label_tr, X_vld, label_vld, X_test, label_test = read_data_atec()  # 读取数据
    y_tr = one_hot(label_tr, 2)
    y_vld = one_hot(label_vld, 2)
    y_test = one_hot(label_test, 2)
    trainer = Trainer(config, X_tr.shape[1])  #训练模型
    trainer.run_epoch(X_tr, y_tr, X_vld, y_vld, X_test, y_test)
    print('CNN-dropout:  Epoch=12, features number:298')

    predict = []
    for x, y in trainer.get_batches(X_test, y_test, config.batch_size*50):
        test_logits = trainer.prediction(x)
        test_logits = list(test_logits)
        predict += test_logits
    rest = len(X_test) % (config.batch_size*50)
    x = X_test[- rest:, :]
    test_logits = trainer.prediction(x)
    test_logits = list(test_logits)
    predict += test_logits
    predict = np.array(predict)
    accuracy = accuracy_score(y_test[:,1], predict)
    precision = precision_score(y_test[:,1], predict)
    recall = recall_score(y_test[:,1], predict)
    f1 = f1_score(y_test[:,1], predict)
    print("Test accuracy: {:.6f}".format(accuracy))
    print("Test precision: {:.6f}".format(precision))
    print("Test recall: {:.6f}".format(recall))
    print("Test F1: {:.6f}".format(f1))

if __name__ == '__main__':
    main_atec(parse_arguments(sys.argv[1:]))
