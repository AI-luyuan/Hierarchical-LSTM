from sentiment import Sentiment
from model_upa import Model
import tensorflow as tf

tf.app.flags.DEFINE_string("task", "np_chunking", "Task.")
tf.app.flags.DEFINE_string("cell", "lstm", "Rnn cell.")
tf.app.flags.DEFINE_integer("size", 200, "Size of each layer.")
tf.app.flags.DEFINE_integer("batch", 16, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 20, "Number of training epoch.")
tf.app.flags.DEFINE_string("loss", "cross_entropy", "Loss function.")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability.")
tf.app.flags.DEFINE_string("gpu", '0', "GPU id.")
tf.app.flags.DEFINE_string("cpu", '0', "CPU id")
tf.app.flags.DEFINE_string("opt",'Adagrad','Optimizer.')

FLAGS = tf.app.flags.FLAGS

data_name = 'yelp13'

def train():
	d = Sentiment(data_name, 5)
	with tf.device('/gpu:'+FLAGS.gpu):
		m = Model(data_name, d.num_class, embeddings = d.embeddings, size = FLAGS.size, batch_size = FLAGS.batch, dropout = FLAGS.dropout,
			rnn_cell = FLAGS.cell, optimize = FLAGS.opt)
		m.fit(d.train_set, d.test_set, FLAGS.epoch)

if __name__ == '__main__':
	train()
