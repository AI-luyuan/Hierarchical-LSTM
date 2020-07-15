import numpy as np
import random
#import h5py
import os
import time
import collections

import pickle as pk 
import os 

class Embeddings(object):
	def __init__(self, data_name, dim=50):
		self.data_name = data_name
		self.word_file = '../data/' + self.data_name + '/wordlist.txt'
		self.word_emb_file = '../data/' + self.data_name + '/embword.save'
		self.user_file = '../data/' + self.data_name + '/usrlist.txt'
		self.user_emb_file = '../data/' + self.data_name + '/embuser.save'
		self.prdt_file = '../data/' + self.data_name + '/prdlist.txt'
		self.prdt_emb_file = '../data/' + self.data_name + '/embprdt.save'
		self.dim = dim

		if os.path.isfile(self.user_emb_file) == False:
			self.build_user_embedding()
		if os.path.isfile(self.prdt_emb_file) == False:
			self.build_prdt_embedding()

		self.word2id, self.word_embeddings = self.load_word_embedding()
		self.user2id, self.user_embeddings = self.load_user_embedding()
		self.prdt2id, self.prdt_embeddings = self.load_prdt_embedding()


	def load_word_embedding(self):
		words = ['UNK']
		for line in open(self.word_file, 'r', encoding='utf-8'):
			word = line.strip()
			words.append(word)
		
		word_embeddings = pk.load(open(self.word_emb_file, 'rb'))
		# print (len(words))
		# print (word_embeddings.shape)
		unk_embedding = np.random.rand(self.dim)*2-1
		word_embeddings = np.insert(word_embeddings, 0, values=unk_embedding, axis=0)

		word2id = dict(zip(words, range(len(words))))
		return word2id, word_embeddings

	def load_user_embedding(self):
		users = ['UNK']
		for line in open(self.user_file, 'r', encoding='utf-8'):
			user = line.strip()
			users.append(user)
		user_embeddings = pk.load(open(self.user_emb_file, 'rb'))
		user2id = dict(zip(users, range(len(users))))
		return user2id, user_embeddings

	def load_prdt_embedding(self):
		prdts = ['UNK']
		for line in open(self.prdt_file, 'r'):
			prdt = line.strip()
			prdts.append(prdt)

		prdt_embeddings = pk.load(open(self.prdt_emb_file, 'rb'))
		prdt2id = dict(zip(prdts, range(len(prdts))))
		return prdt2id, prdt_embeddings


	def build_user_embedding(self):
		users = []
		for line in open(self.user_file, 'r', encoding='utf-8'):
			user = line.strip()
			users.append(user)

		n_usr = len(users)
		user_embeddings = np.asarray(
			np.random.normal(scale=0.1, size=(n_usr+1, self.dim)), 
			dtype=np.float32)
		# user_embeddings = np.zeros((n_usr+1,self.dim),dtype=np.float32)
		pk.dump(user_embeddings, open(self.user_emb_file, 'wb'))
		return


	def build_prdt_embedding(self):
		prdts = []
		for line in open(self.prdt_file, 'r'):
			prdt = line.strip()
			prdts.append(prdt)

		n_prd = len(prdts)

		prdt_embeddings = np.asarray(
			np.random.normal(scale=0.1, size=(n_prd+1, self.dim)), 
			dtype=np.float32)

		pk.dump(prdt_embeddings, open(self.prdt_emb_file, 'wb'))
		return

	def docs2ids(self, docs):
		id_set = []
		for doc in docs:
			id_set.append([])
			for sentence in doc:
				id_set[-1].append([])
				for word in sentence:
					if word not in self.word2id:
						id_set[-1][-1].append(self.word2id['UNK'])
					else:
						id_set[-1][-1].append(self.word2id[word])
		return id_set

	def users2ids(self, users):
		id_set = []
		for user in users:
			if user not in self.user2id:
				id_set.append(self.user2id['UNK'])
			else:
				id_set.append(self.user2id[user])
		return id_set

	def prdts2ids(self, prdts):
		id_set = []
		for prdt in prdts:
			if prdt not in self.prdt2id:
				id_set.append(self.prdt2id['UNK'])
			else:
				id_set.append(self.prdt2id[prdt])
		return id_set

