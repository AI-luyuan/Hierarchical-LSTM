import numpy as np
import random
#import h5py
import os
import time
import collections
from embeddings import Embeddings

class Sentiment(object):
	"""NP_chunking data preparation"""
	def __init__(self, data_name, num_class=5):
		self.data_name = data_name
		self.train_data_path = '../data/' + self.data_name + '/train.txt'
		self.test_data_path = '../data/' + self.data_name + '/test.txt'
		self.dev_data_path = '../data/' + self.data_name + '/dev.txt'
		self.embeddings = Embeddings(data_name)
		self.num_class = num_class
		start_time = time.time()
		self.load_data()
		print ('Reading datasets comsumes %.3f seconds' % (time.time()-start_time))
			
	def deal_with_data(self, path):
		users, products, labels, docs, len_docs, len_words = [], [], [], [], [], []
		k = 0
		for line in open(path, 'r', encoding='UTF-8'):
			tokens = line.strip().split('\t\t')
			users.append(tokens[0])
			products.append(tokens[1])
			labels.append(int(tokens[2])-1)
			doc = tokens[3].strip().split('<sssss>')
			len_docs.append(len(doc))
			doc = [sentence.strip().split(' ') for sentence in doc]
			len_words.append([len(sentence) for sentence in doc])
			docs.append(doc)
			k += 1
		return users, products, labels, docs

	def load_data(self):
		train_users, train_products, train_labels, train_docs = self.deal_with_data(self.train_data_path)
		test_users, test_products, test_labels, test_docs = self.deal_with_data(self.test_data_path)
		dev_users, dev_products, dev_labels, dev_docs = self.deal_with_data(self.dev_data_path)


		train_docs = self.embeddings.docs2ids(train_docs)
		test_docs = self.embeddings.docs2ids(test_docs)
		dev_docs = self.embeddings.docs2ids(dev_docs)

		train_users = self.embeddings.users2ids(train_users)
		test_users = self.embeddings.users2ids(test_users)
		dev_users = self.embeddings.users2ids(dev_users)

		train_products = self.embeddings.prdts2ids(train_products)
		test_products = self.embeddings.prdts2ids(test_products)
		dev_products = self.embeddings.prdts2ids(dev_products)

		self.train_set = list(zip(train_docs, train_labels, train_users, train_products))
		self.test_set = list(zip(test_docs, test_labels, test_users, test_products))
		self.dev_set = list(zip(dev_docs, dev_labels, dev_users, dev_products))

