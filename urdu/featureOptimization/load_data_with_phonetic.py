import os
from nltk import FreqDist
import numpy as np
import re
import datetime
import sys
import gc
import pickle 

from collections import Counter, deque

MAX_LEN = 20
VOCAB_SIZE = 65
VOCAB_SIZE_WORDS = 30000

def getIndexedWords(X_unique, y_unique, orig=False, test=False):
	X_un = [list(x) for x,w in zip(X_unique, y_unique) if len(x) > 0 and len(w) > 0]

	X = X_un
	print("X:", X[:10])
	# build a vocabulary of most frequent characters
	dist = FreqDist(np.hstack(X))
	X_vocab = dist.most_common(92)


	print("######### Remove erroneous characters ##########")
	for i in X_vocab:
		if i[0] == '\u200d' or i[0] == '\u200b':
			X_vocab.remove(i)

	if test == True:
		X_word2idx = pickle.load(open('./pickle_dumps/X_word2idx', 'rb'))
		X_idx2word = pickle.load(open('./pickle_dumps/X_idx2word', 'rb'))

	else:
		X_idx2word = [letter[0] for letter in X_vocab]
		X_idx2word.insert(0, 'Z') # 'Z' is the starting token
		X_idx2word.append('U') # 'U' for out-of-vocab characters
		
		# create letter-to-index mapping
		X_word2idx =  {letter:idx for idx, letter in enumerate(X_idx2word)}

	for i, word in enumerate(X):
		for j, char in enumerate(word):
			if char in X_word2idx:
				X[i][j] = X_word2idx[char]
			else:
				X[i][j] = X_word2idx['U']

	if orig == True:
		return X, X_un, X_vocab, X_word2idx, X_idx2word
	else:
		return X

def getSentenceWiseAdjustedRight(l1, l2):
	newlist = []
	toBeRemoved=False

	for i, j in zip(l1, l2):
		if toBeRemoved == True:
			newlist += ' '
		elif toBeRemoved == False:
			newlist += j
		if j == '|' or j == '?' or j == '!':
			toBeRemoved = True
		if i == '|' or i == '?' or i == '!':
			toBeRemoved = False

	return newlist

def getSentenceWiseAdjustedLeft(l1, l2):
	newlist = []
	toBeRemoved=False

	for i, j in zip(l1, l2):
		if toBeRemoved == True:
			newlist += ' '
		elif toBeRemoved == False:
			newlist += j
		if i == '|' or i == '?' or i == '!':
			toBeRemoved = True
		if j == '|' or j == '?' or j == '!':
			toBeRemoved = False

	return newlist

def load_data_for_features(features):
	pos = []; gender = []; num = []; person = []; case = []; tam = []
	
	for sentence in features:
		pos += [word[0] for word in sentence]
		gender += [word[1] for word in sentence]
		num += [word[2] for word in sentence]
		person += [word[3] for word in sentence]
		case += [word[4] for word in sentence]
		tam += [word[5] for word in sentence]
	
	return (pos, gender, num, person, case, tam)


def load_data_for_seq2seq(sentences, rootwords, X_phonetic=None, features=None, labels=None, test=False, context1=False, context2=False, context3=False,
	context4=False, context5=False):
	#print(sentences[:2])
	
	X_unique = [item for sublist in sentences for item in sublist]
	y_unique = [item for sublist in rootwords for item in sublist]


	############## processing of test set ################
	if features != None:
 		j = 0
 		y1,y2,y3,y4,y5,y6 = features
 		l1, l2, l3, l4, l5, l6 = labels

 		X_phonetic = X_phonetic.tolist() # numpy arrays don't support deletion
 		complete_list = [X_unique, X_phonetic, y_unique, y1, y2, y3, y4, y5, y6]

 		copy = X_unique

 		cnt = len(X_unique)
 		i = 0
 		while i < cnt:
 			#try:
	 		if y1[i] not in l1 or y2[i] not in l2 or y3[i] not in l3 or y4[i] not in l4 \
	 		or y5[i] not in l5 or y6[i] not in l6 :
	 			for item in complete_list:
	 				print("Deleting element:",j)
	 				j += 1
 					del item[i]
 					cnt = cnt - 1
 				i = i - 1
 			i = i + 1

 		X_phonetic = np.asarray(X_phonetic)
	#####################################################

	# process vocab indexing for X in the function since we will need to call it multiple times
	X, X_un, X_vocab, X_word2idx, X_idx2word = getIndexedWords(X_unique, y_unique, orig=True, test=test)

	if test == False:
		pickle.dump(X_word2idx, open('./pickle_dumps/X_word2idx', 'wb'))
		pickle.dump(X_idx2word, open('./pickle_dumps/X_idx2word', 'wb'))
	else:
		# pickle.dump(removed_indices, open('./pickle_dumps/removed_indices', 'wb'))
		X_word2idx = pickle.load(open('./pickle_dumps/X_word2idx', 'rb'))

	# process vocab indexing for y here, since only single processing required
	y_un = [list(w) for x,w in zip(X_unique, y_unique) if len(x) > 0 and len(w) > 0]	
	y = y_un
	
	for i, word in enumerate(y):
		for j, char in enumerate(word):
			if char in X_word2idx:
				y[i][j] = X_word2idx[char]
			else:
				y[i][j] = X_word2idx['U']
	
	# consider a context of 1 word right and left each
	# make two lists by shifting the elements
	if context1 == True or context2 == True or context3 == True or context4 == True or context5 == True: 

		X_left = deque(X_unique)
		
		X_left.append(' ') # all elements would be shifted one left
		X_left.popleft()
		X_left1 = list(X_left)
		X_left1 = getIndexedWords(X_left1, y_unique, orig=False, test=test)

		X_left.append(' ')
		X_left.popleft()
		X_left2 = list(X_left)
		X_left2 = getIndexedWords(X_left2, y_unique, orig=False, test=test)
		
		X_left.append(' ')
		X_left.popleft()
		X_left3 = list(X_left)
		X_left3 = getIndexedWords(X_left3, y_unique, orig=False, test=test)

		X_left.append(' ')
		X_left.popleft()
		X_left4 = list(X_left)
		X_left4 = getIndexedWords(X_left4, y_unique, orig=False, test=test)	

		X_left.append(' ')
		X_left.popleft()
		X_left5 = list(X_left)
		X_left5 = getIndexedWords(X_left5, y_unique, orig=False, test=test)	

		X_right_orig = X_unique
		X_right = deque(X_right_orig)

		X_right.appendleft(' ') 
		X_right.pop()
		X_right1 = list(X_right)
		X_right1 = getIndexedWords(X_right1, y_unique, orig=False, test=test)

		X_right.appendleft(' ')
		X_right.pop()
		X_right2 = list(X_right)
		X_right2 = getIndexedWords(X_right2, y_unique, orig=False, test=test)

		X_right.appendleft(' ')
		X_right.pop()
		X_right3 = list(X_right)
		X_right3 = getIndexedWords(X_right3, y_unique, orig=False, test=test)

		X_right.appendleft(' ')
		X_right.pop()
		X_right4 = list(X_right)
		X_right4 = getIndexedWords(X_right4, y_unique, orig=False, test=test)

		X_right.appendleft(' ')
		X_right.pop()
		X_right5 = list(X_right)
		X_right5 = getIndexedWords(X_right5, y_unique, orig=False, test=test)

		print(len(X_left1))
		print(len(X_left2))
		print(len(X_right1))
		print(len(X_right2))

		if context1 == True:
			if test == True:
				complete_list = [X_un, y_un, y1, y2, y3, y4, y5, y6]
				return (complete_list, X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, X_word2idx, X_idx2word, X_left, X_right, X_phonetic)
			else:
				return (X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, X_word2idx, X_idx2word, X_left, X_right)

		elif context2 == True:
			if test == True:
				complete_list = [X_un, y_un, y1, y2, y3, y4, y5, y6]
				return (complete_list, X, len(X_vocab)+2, X_word2idx, X_idx2word, y, 
					len(X_vocab)+2, X_word2idx, X_idx2word, X_left1, X_left2, X_right1, X_right2, X_phonetic)
			else:
				return (X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, 
					X_word2idx, X_idx2word, X_left1, X_left2, X_right1, X_right2)

		elif context3 == True:
			if test == True:
				complete_list = [X_un, y_un, y1, y2, y3, y4, y5, y6]
				return (complete_list, X, len(X_vocab)+2, X_word2idx, X_idx2word, y, 
					len(X_vocab)+2, X_word2idx, X_idx2word, X_left1, X_left2, X_left3, X_right1, X_right2, X_right3, X_phonetic)
			else:
				return (X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, 
					X_word2idx, X_idx2word, X_left1, X_left2, X_left3, X_right1, X_right2, X_right3) 

		elif context4 == True:
			if test == True:
				complete_list = [X_un, y_un, y1, y2, y3, y4, y5, y6]
				return (complete_list, X, len(X_vocab)+2, X_word2idx, X_idx2word, y, 
					len(X_vocab)+2, X_word2idx, X_idx2word, X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4, X_phonetic)
			else:
				return (X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, 
					X_word2idx, X_idx2word, X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4) 

		elif context5 == True:
			if test == True:
				complete_list = [X_un, y_un, y1, y2, y3, y4, y5, y6]
				return (complete_list, X, len(X_vocab)+2, X_word2idx, X_idx2word, y, 
					len(X_vocab)+2, X_word2idx, X_idx2word, X_left1, X_left2, X_left3, X_left4, X_left5, X_right1, \
					 X_right2, X_right3, X_right4, X_right5, X_phonetic)
			else:
				return (X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, 
					X_word2idx, X_idx2word, X_left1, X_left2, X_left3, X_left4, X_left5, X_right1, X_right2, X_right3, X_right4, X_right5) 
	else:
		if test == True:
			complete_list = [X_un, y_un, y1, y2, y3, y4, y5, y6]
			return (complete_list, X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, X_word2idx, X_idx2word, X_phonetic)
		else:
			return (X, len(X_vocab)+2, X_word2idx, X_idx2word, y, len(X_vocab)+2, X_word2idx, X_idx2word)

# To preserve the sentences without flatting out the lists
def load_data_for_features_sentencewise(features):
	# this function is different from above two in the sense that
	# the vocabulary is built on a word level instead of character levels.	
	splitted_feature = [] # seggregate all the features
	for feature in features:
		this_sentence = []
		for i in feature:
			this_sentence.append(i.split("|"))
		splitted_feature.append(this_sentence)

	#print(splitted_feature[8:10])
	
	#print(len(splitted_feature))
	
	y1 = []
	y2 = []
	y3 = []
	y4 = []
	y5 = []
	y6 = []
	y7 = []
	y8 = []

	all_features = [y1, y2, y3, y4, y5, y6, y7, y8]

	for feature in splitted_feature:
		f1 = []
		f2 = []
		f3 = []
		f4 = []
		f5 = []
		f6 = []
		f7 = []
		f8 = []

		all_fs = [f1, f2, f3, f4, f5, f6, f7, f8]
		for this in feature:
			for i,j in zip(this[:8], all_fs):
				val = re.sub(r'.*-', '', i) 
				if len(val) != 0:
					j.append(val)
				else:
					j.append('Unk')

		for i,j in zip(all_features, all_fs):
			i.append(j)

	#print(y2[8:10])

	pickle.dump(y1, open('y1_test', 'wb'))
	pickle.dump(y2, open('y2_test', 'wb'))
	pickle.dump(y3, open('y3_test', 'wb'))

