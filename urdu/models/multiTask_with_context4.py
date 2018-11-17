import pickle
import operator
import tensorflow as tf 
from load_data import load_data_for_seq2seq, load_data_for_features

import keras.backend as K
from keras.utils import np_utils
from keras import initializers, regularizers, constraints
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import dot, Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Input, merge, concatenate
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Layer
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints
from attention_decoder import AttentionDecoder 
from attention_encoder import AttentionWithContext

from nltk import FreqDist
import numpy as np
import pandas as pd 
import os
import datetime
import sys
import gc 
import bisect # for searching through label encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, f1_score
from collections import Counter, deque
from predict_with_features import plot_model_performance
#from curve_plotter import plot_precision_recall

MODE='test'

HIDDEN_DIM = 40
EPOCHS = 200
dropout = 0.2
TIME_STEPS = 20
EMBEDDING_DIM = 150
BATCH_SIZE = 100
LAYER_NUM = 2

class_labels = []


def write_words_to_file(orig_words, predictions):
	print("Writing to file ..")
	#print(sentences[:10])

	sentences = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))

	X = [item for sublist in sentences for item in sublist]
	Y = [item for sublist in orig_words for item in sublist]

	filename = "./outputs/BLSTM_with_context4_and_att/multitask_context_out.txt"
	with open(filename, 'w', encoding='utf-8') as f:
		f.write("Words" + '\t\t\t' + 'Original Roots' + '\t\t' + "Predicted roots" + '\n')
		for a,b,c in zip(X, Y, predictions):
			f.write(str(a) + '\t\t' + str(b) + '\t\t\t' + str(c) + '\n')

	print("Success writing to file !")

def write_features_to_file(orig_features, pred_features, encoders):

	orig_features[:] = [ [np.where(r == 1)[0][0] for r in x] for x in orig_features]
	print(orig_features[0][:10])
	pred_features[:] = [x.tolist() for x in pred_features]
	
	for i in range(len(orig_features)):
		orig_features[i] = encoders[i].inverse_transform(orig_features[i])
		print(orig_features[i][:10])
		pred_features[i] = encoders[i].inverse_transform(pred_features[i])
		print(pred_features[i][:10])

	sentences = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))
	words = [item for sublist in sentences for item in sublist]

	for i in range(len(orig_features)):
		filename = "./outputs/BLSTM_with_context4_and_att/feature"+str(i)+"context_out.txt"
		with open(filename, 'w', encoding='utf-8') as f:
			f.write("Word" + '\t\t' + 'Original feature' + '\t' + 'Predicted feature' + '\n')
			for a,b,c in zip(words, orig_features[i], pred_features[i]):
				f.write(str(a) + '\t\t' + str(b) + '\t\t\t' + str(c) + '\n')

	print("Success writing features to files !!")

def process_features(y1,y2,y3,y4,y5,y7,y8, n = None, enc=None):

	y = [y1, y2, y3, y4, y5, y7, y8]

	print(y1[:10])

	in_cnt1 = Counter(y1)
	in_cnt2 = Counter(y2)
	in_cnt3 = Counter(y3)
	in_cnt4 = Counter(y4)
	in_cnt5 = Counter(y5)
	in_cnt6 = Counter(y7)
	in_cnt7 = Counter(y8)

	labels=[] # for processing of unnecessary labels from the test set
	init_cnt = [in_cnt1, in_cnt2, in_cnt3, in_cnt4, in_cnt5, in_cnt6, in_cnt7]

	for i in range(len(init_cnt)):
		labels.append(list(init_cnt[i].keys()))

	if enc == None:
		enc = {}
		transformed = []
		print("processing train encoders!")
		for i in range(len(y)):
			enc[i] = LabelEncoder()
			transformed.append(enc[i].fit_transform(y[i]))

	else:
		transformed = []
		print("processing test encoders !")
		for i in range(len(y)):
			#y[i] = list(map(lambda s: '<unk>' if s not in enc[i].classes_ else s, y[i]))
			#enc_classes = enc[i].classes_.tolist()
			#bisect.insort_left(enc_classes, '<unk>')
			#enc[i].classes_ = enc_classes
			transformed.append(enc[i].transform(y[i]))

	y1 = list(transformed[0])
	y2 = list(transformed[1])
	y3 = list(transformed[2])
	y4 = list(transformed[3])
	y5 = list(transformed[4])
	y7 = list(transformed[5])
	y8 = list(transformed[6])

	print(y1[:10])

	cnt1 = Counter(y1)
	cnt2 = Counter(y2)
	cnt3 = Counter(y3)
	cnt4 = Counter(y4)
	cnt5 = Counter(y5)
	cnt6 = Counter(y7)
	cnt7 = Counter(y8)

	if enc != None:
		lis = [cnt1, cnt2, cnt3, cnt4, cnt5, cnt6, cnt7]
		for i in range(len(lis)):
			class_labels.append(list(lis[i].keys()))


	print(format(cnt1))
	print(format(cnt2))
	print(format(cnt3))

	if n == None:
		n1 = max(cnt1, key=int) + 1
		n2 = max(cnt2, key=int) + 1
		n3 = max(cnt3, key=int) + 1
		n4 = max(cnt4, key=int) + 1
		n5 = max(cnt5, key=int) + 1
		n6 = max(cnt6, key=int) + 1
		n7 = max(cnt7, key=int) + 1
	
	else:
		n1,n2,n3,n4,n5,n6,n7 = n

	y1 = np_utils.to_categorical(y1, num_classes=n1)
	y2 = np_utils.to_categorical(y2, num_classes=n2)
	y3 = np_utils.to_categorical(y3, num_classes=n3)
	y4 = np_utils.to_categorical(y4, num_classes=n4)
	y5 = np_utils.to_categorical(y5, num_classes=n5)
	y7 = np_utils.to_categorical(y7, num_classes=n6)
	y8 = np_utils.to_categorical(y8, num_classes=n7)

	return (y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n6, y8, n7, enc, labels)

def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6, hidden_size, num_layers):

	def smart_merge(vectors, **kwargs):
			return vectors[0] if len(vectors)==1 else merge(vectors, **kwargs)		
	
	current_word = Input(shape=(X_max_len,), dtype='int32')
	right_word1 = Input(shape=(X_max_len,), dtype='int32')
	right_word2 = Input(shape=(X_max_len,), dtype='int32')
	right_word3 = Input(shape=(X_max_len,), dtype='int32')
	right_word4 = Input(shape=(X_max_len,), dtype='int32')
	left_word1 = Input(shape=(X_max_len,), dtype='int32')
	left_word2 = Input(shape=(X_max_len,), dtype='int32')
	left_word3 = Input(shape=(X_max_len,), dtype='int32')
	left_word4 = Input(shape=(X_max_len,), dtype='int32')

	emb_layer = Embedding(X_vocab_len, EMBEDDING_DIM, 
				input_length=X_max_len,
				mask_zero=True) 
	
	current_word_embedding = emb_layer(current_word) # POSITION of layer
	right_word_embedding1 = emb_layer(right_word1) # these are the left shifted X by 1
	right_word_embedding2 = emb_layer(right_word2) # left shifted by 2
	right_word_embedding3 = emb_layer(right_word3)
	right_word_embedding4 = emb_layer(right_word4)

	left_word_embedding1 = emb_layer(left_word1) # these are the right shifted X by 1, i.e. the left word is at current position
	left_word_embedding2 = emb_layer(left_word2)
	left_word_embedding3 = emb_layer(left_word3)
	left_word_embedding4 = emb_layer(left_word4)

	BidireLSTM_curr= Bidirectional(LSTM(40, dropout=dropout, return_sequences=True))(current_word_embedding)
	BidireLSTM_right1 = Bidirectional(LSTM(40, dropout=dropout, return_sequences=True))(right_word_embedding1)
	BidireLSTM_right2 = Bidirectional(LSTM(40, dropout=dropout, return_sequences=True))(right_word_embedding2)
	BidireLSTM_right3 = Bidirectional(LSTM(40, dropout=dropout, return_sequences=True))(right_word_embedding3)
	BidireLSTM_right4 = Bidirectional(LSTM(40, dropout=dropout, return_sequences=True))(right_word_embedding4)

	BidireLSTM_left1 = Bidirectional(LSTM(40, dropout=dropout, return_sequences=True))(left_word_embedding1)
	BidireLSTM_left2 = Bidirectional(LSTM(40, dropout=dropout, return_sequences=True))(left_word_embedding2)
	BidireLSTM_left3 = Bidirectional(LSTM(40, dropout=dropout, return_sequences=True))(left_word_embedding3)
	BidireLSTM_left4 = Bidirectional(LSTM(40, dropout=dropout, return_sequences=True))(left_word_embedding4)

	att = AttentionWithContext()(BidireLSTM_curr)
	#print(att.shape)
	RepLayer= RepeatVector(y_max_len)
	RepVec= RepLayer(att)
	Emb_plus_repeat=[current_word_embedding]
	Emb_plus_repeat.append(RepVec)
	Emb_plus_repeat = smart_merge(Emb_plus_repeat, mode='concat')
	
	
	for _ in range(num_layers):
		LtoR_LSTM = Bidirectional(LSTM(40, dropout=dropout, return_sequences=True))
		temp = LtoR_LSTM(Emb_plus_repeat)
	
	# for each time step in the input, we intend to output |y_vocab_len| time steps
	time_dist_layer = TimeDistributed(Dense(y_vocab_len))(temp)
	outputs = Activation('softmax')(time_dist_layer)
	
	# Only for the tags prediction, will we be requiring the context words
	concatenated_encodings = [BidireLSTM_curr]
	concatenated_encodings.append(BidireLSTM_left1)
	concatenated_encodings.append(BidireLSTM_right1)
	concatenated_encodings.append(BidireLSTM_left2)
	concatenated_encodings.append(BidireLSTM_right2)
	concatenated_encodings.append(BidireLSTM_left3)
	concatenated_encodings.append(BidireLSTM_right3)
	concatenated_encodings.append(BidireLSTM_left4)
	concatenated_encodings.append(BidireLSTM_right4)

	concatenated_encodings = smart_merge(concatenated_encodings, mode='concat')

	att2 = AttentionWithContext()(concatenated_encodings)

	RepVec= RepLayer(att2)
	Emb_plus_repeat=[current_word_embedding]
	Emb_plus_repeat.append(RepVec)
	Emb_plus_repeat = smart_merge(Emb_plus_repeat, mode='concat')
	
	BidireLSTM_vector = Bidirectional(LSTM(40, dropout=dropout, return_sequences=False))(Emb_plus_repeat)
	
	out1 = Dense(n1, activation='softmax')(BidireLSTM_vector)	
	out2 = Dense(n2, activation='softmax')(BidireLSTM_vector)	
	out3 = Dense(n3, activation='softmax')(BidireLSTM_vector)	
	out4 = Dense(n4, activation='softmax')(BidireLSTM_vector)	
	out5 = Dense(n5, activation='softmax')(BidireLSTM_vector)	
	out6 = Dense(n6, activation='softmax')(BidireLSTM_vector)	

	all_inputs = [current_word, right_word1, right_word2, right_word3, right_word4, left_word1, left_word2, left_word3, left_word4]
	all_outputs = [outputs, out1, out2, out3, out4, out5, out6]

	model = Model(input=all_inputs, output=all_outputs)
	opt = Adam()
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], 
		loss_weights=[1., 1., 1., 1., 1., 1., 1.])
	
	return model

def process_data(word_sentences, max_len, word_to_ix):
	# Vectorizing each element in each sequence
	sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
	for i, sentence in enumerate(word_sentences):
		for j, word in enumerate(sentence):
			sequences[i, j, word] = 1
	return sequences

sentences = pickle.load(open('./pickle-dumps/sentences_intra', 'rb'))
rootwords = pickle.load(open('./pickle-dumps/rootwords_intra', 'rb'))
features = pickle.load(open('./pickle-dumps/features_intra', 'rb'))

# we keep X_idx2word and y_idx2word the same
# X_left & X_right = X shifted to one and two positions left and right for context2
X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word, X_left1, X_left2, X_left3, X_left4, \
	X_right1, X_right2, X_right3, X_right4 = load_data_for_seq2seq(sentences, rootwords, test=False, context4=True)

y1, y2, y3, y4, y5, y6, y7, y8 = load_data_for_features(features)

y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n7, y8, n8, enc, labels = process_features(y1, y2, y3, y4, y5, y7, y8)

n = [n1, n2, n3, n4, n5, n7, n8]
print(labels)

# should be all equal for better results
print(len(X))
print(X_vocab_len)
print(len(X_word_to_ix))
print(len(X_ix_to_word))
#print(len(y_word_to_ix))
print(len(y_ix_to_word))


X_max = max([len(word) for word in X])
y_max = max([len(word) for word in y])
X_max_len = max(X_max,y_max)
y_max_len = max(X_max,y_max)

print(X_max_len)
print(y_max_len)

print("Zero padding .. ")
X = pad_sequences(X, maxlen= X_max_len, dtype = 'int32', padding='post')
X_left1 = pad_sequences(X_left1, maxlen = X_max_len, dtype='int32', padding='post')
X_left2 = pad_sequences(X_left2, maxlen = X_max_len, dtype='int32', padding='post')
X_left3 = pad_sequences(X_left3, maxlen = X_max_len, dtype='int32', padding='post')
X_left4 = pad_sequences(X_left4, maxlen = X_max_len, dtype='int32', padding='post')
X_right1 = pad_sequences(X_right1, maxlen = X_max_len, dtype='int32', padding='post')
X_right2 = pad_sequences(X_right2, maxlen = X_max_len, dtype='int32', padding='post')
X_right3 = pad_sequences(X_right3, maxlen = X_max_len, dtype='int32', padding='post')
X_right4 = pad_sequences(X_right4, maxlen = X_max_len, dtype='int32', padding='post')
y = pad_sequences(y, maxlen = y_max_len, dtype = 'int32', padding='post')

print("Compiling Model ..")
model = create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len,
		y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n7, HIDDEN_DIM, LAYER_NUM)

saved_weights = "./model_weights/multiTask_with_context4_att.hdf5"

if MODE == 'train':
	print("Training model ..")
	y_sequences = process_data(y, y_max_len, y_word_to_ix)

	hist = model.fit([X, X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4], [y_sequences, y1, y2, y3, y4, y5, y7], 
		validation_split=0.25, 
		batch_size=BATCH_SIZE, epochs=EPOCHS,  
		callbacks=[EarlyStopping(patience=7),
		ModelCheckpoint('./model_weights/multiTask_with_context4_att.hdf5', save_best_only=True,
			verbose=1)])

	print(hist.history.keys())
	print(hist)
	plot_model_performance(
		train_loss=hist.history.get('loss', []),
	    train_acc=hist.history.get('acc', []),
	    train_val_loss=hist.history.get('val_loss', []),
	    train_val_acc=hist.history.get('val_acc', [])
	)

																																																																									
else:
	if len(saved_weights) == 0:
		print("network hasn't been trained!")
		sys.exit()
	else:
		test_sample_num = 0

		test_sentences = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))
		test_roots = pickle.load(open('./pickle-dumps/rootwords_test', 'rb'))
		test_features = pickle.load(open('./pickle-dumps/features_test', 'rb'))
		
		y1, y2, y3, y4, y5, y6, y7, y8 = load_data_for_features(test_features)
		features = [y1, y2, y3, y4, y5, y7, y8]

		complete_list,X_test, X_vcab_len, X_wrd_to_ix, X_ix_to_wrd, y_test, y_vcab_len, y_wrd_to_ix, y_ix_to_wrd, \
		X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4 = \
				load_data_for_seq2seq(test_sentences, test_roots, features, labels, test=True, context4=True)

		X_orig, y_orig, y1, y2, y3, y4, y5, y7, y8 = complete_list

		X_test = pad_sequences(X_test, maxlen=X_max_len, dtype='int32', padding='post')
		X_left1 = pad_sequences(X_left1, maxlen=X_max_len, dtype='int32', padding='post')
		X_right1 = pad_sequences(X_right1, maxlen=X_max_len, dtype='int32', padding='post')
		X_left2 = pad_sequences(X_left2, maxlen=X_max_len, dtype='int32', padding='post')
		X_right2 = pad_sequences(X_right2, maxlen=X_max_len, dtype='int32', padding='post')
		X_left3 = pad_sequences(X_left3, maxlen=X_max_len, dtype='int32', padding='post')
		X_right3 = pad_sequences(X_right3, maxlen=X_max_len, dtype='int32', padding='post')
		X_left4 = pad_sequences(X_left4, maxlen=X_max_len, dtype='int32', padding='post')
		X_right4 = pad_sequences(X_right4, maxlen=X_max_len, dtype='int32', padding='post')
		y_test = pad_sequences(y_test, maxlen=X_max_len, dtype='int32', padding='post')

		y_test_seq = process_data(y_test, y_max_len, y_word_to_ix)

		y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n7, y8, n8, enc, lab = process_features(y1, y2, y3, y4, y5, y7, y8, n, enc) # pass previous encoders as args

		model.load_weights(saved_weights)

		plot_model(model, to_file="multi_task_arch_with_context4.png", show_shapes=True)

		print(model.evaluate([X_test, X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4], [y_test_seq, y1, y2, y3, y4, y5, y7]))
		print(model.metrics_names)
		
		words, f1, f2, f3, f4, f5, f7 = model.predict([X_test, X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4])
		
		predictions = np.argmax(words, axis=2)

		# uncomment these when need to write features to text file
		# comment these for pickle dump 
		'''
		f1 = np.argmax(f1, axis=1)
		f2 = np.argmax(f2, axis=1)
		f3 = np.argmax(f3, axis=1)
		f4 = np.argmax(f4, axis=1)
		f5 = np.argmax(f5, axis=1)
		f7 = np.argmax(f7, axis=1)
		'''
		pred_features = [f1, f2, f3, f4, f5, f7]
		orig_features = [y1, y2, y3, y4, y5, y7]
		
		# Note: Either pickle dump or human-apprehensible outputs 
		
		pickle.dump(pred_features, open('./pickle-dumps/predictions_context4_att', 'wb'))
		pickle.dump(orig_features, open('./pickle-dumps/originals', 'wb'))
		pickle.dump(n, open('./pickle-dumps/num_classes', 'wb'))
		pickle.dump(class_labels, open('./pickle-dumps/class_labels', 'wb'))
		'''
		# uncomment for generating human-apprehensible output files
		######### Post processing of the features ############
		write_features_to_file(orig_features, pred_features, enc)
		######################################################
		
		######### Post processing of predicted roots ##############
		sequences = []

		for i in predictions:
			test_sample_num += 1

			char_list = []
			for idx in i:
				if idx > 0:
					char_list.append(y_ix_to_word[idx])

			sequence = ''.join(char_list)
			#print(test_sample_num,":", sequence)
			sequences.append(sequence)

		write_words_to_file(test_roots, sequences)

		'''

