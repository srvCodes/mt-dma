import pickle
import operator
import tensorflow as tf 
from load_data import load_data_for_seq2seq, load_test_data, load_data_for_features

import keras.backend as K
from keras.utils import np_utils
from keras import initializers, regularizers, constraints
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import dot, Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Input, merge
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Layer
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.engine.topology import Layer, InputSpec
from keras import initializers, regularizers, constraints
from attention_decoder import AttentionDecoder 

from nltk import FreqDist
import numpy as np
import os
import datetime
import sys
import gc 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter


MODE='train'

HIDDEN_DIM = 40
EPOCHS = 200
dropout = 0.2
TIME_STEPS = 20
EMBEDDING_DIM = 150
BATCH_SIZE = 30
LAYER_NUM = 2

def create_model(X_vocab_len, X_max_len, y1, num_classes, hidden_size, num_layers):

	def smart_merge(vectors, **kwargs):
			return vectors[0] if len(vectors)==1 else merge(vectors, **kwargs)		
	
	root_word_in = Input(shape=(X_max_len,), dtype='int32')
	
	emb_layer = Embedding(X_vocab_len, EMBEDDING_DIM, 
				input_length=X_max_len,
				mask_zero=True) 
	
	hindi_word_embedding = emb_layer(root_word_in) # POSITION of layer

	BidireLSTM_vector= Bidirectional(LSTM(40, dropout=0.2, return_sequences=False))(hindi_word_embedding)
	outputs = Dense(num_classes, activation='softmax')(BidireLSTM_vector)	
	
	all_inputs = [root_word_in]
	model = Model(input=all_inputs, output=outputs)
	opt = Adam()
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	
	return model

def process_data(word_sentences, max_len, word_to_ix):
	# Vectorizing each element in each sequence
	sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
	for i, sentence in enumerate(word_sentences):
		for j, word in enumerate(sentence):
			sequences[i, j, word] = 1
	return sequences

sentences = pickle.load(open('sentences_intra', 'rb'))
rootwords = pickle.load(open('rootwords_intra', 'rb'))
features = pickle.load(open('features_intra', 'rb'))

# we keep X_idx2word and y_idx2word the same
X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word = load_data_for_seq2seq(sentences, rootwords)
y1, y2, y3, y4, y5, y6, y7, y8 = load_data_for_features(sentences, features)

# should be all equal for better results
print(len(X))
print(X_vocab_len)
print(len(X_word_to_ix))
print(len(X_ix_to_word))
print(len(y_word_to_ix))
print(len(y_ix_to_word))


X_max = max([len(word) for word in X])
y_max = max([len(word) for word in y])
X_max_len = max(X_max,y_max)
y_max_len = max(X_max,y_max)

print(X_max_len)
print(y_max_len)

print("Zero padding .. ")
X = pad_sequences(X, maxlen= X_max_len, dtype = 'int32', padding='post')

# convert y1..y8 into labels
enc1 = LabelEncoder()
enc2 = LabelEncoder()
enc3 = LabelEncoder()
enc4 = LabelEncoder()
enc5 = LabelEncoder()
enc6 = LabelEncoder()
enc7 = LabelEncoder()
enc8 = LabelEncoder()

y1 = enc1.fit_transform(y1)
y2 = enc2.fit_transform(y2)
y3 = enc2.fit_transform(y3)
y4 = enc2.fit_transform(y4)
y5 = enc2.fit_transform(y5)
y6 = enc2.fit_transform(y6)
y7 = enc2.fit_transform(y7)
y8 = enc2.fit_transform(y8)

cnt = Counter(y2)
print(format(cnt))
num_classes = max(cnt, key=int) + 1


y2 = np_utils.to_categorical(y2, num_classes=num_classes)

print(y3[:10])

print("Compiling Model ..")
model = create_model(X_vocab_len, X_max_len, y2, num_classes, HIDDEN_DIM, LAYER_NUM)

saved_weights = "y1.hdf5"

if MODE == 'train':
	print("Training model ..")
	
	history = model.fit(X, y2, validation_split=0.2, 
		batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, 
		callbacks=[EarlyStopping(patience=10, verbose=1),
		ModelCheckpoint('y2.hdf5', save_best_only=True,
			verbose=1)])
	print(history.history.keys())
	print(history)
'''
else:
	if len(saved_weights) == 0:
		print("network hasn't been trained!")
		sys.exit()
	else:
		test_sample_num = 0

		test_sentences = pickle.load(open('sentences_test', 'rb'))
		test_roots = pickle.load(open('rootwords_test', 'rb'))
		test_features = pickle.load(open('features_test', 'rb'))
		
		X_test, X_unique, y_unique = load_test_data(test_sentences, test_roots, X_word_to_ix)

		X_test = pad_sequences(X_test, maxlen=X_max_len, dtype='int32', padding='post')
		
		model.load_weights(saved_weights)

		plot_model(model, to_file="model2_arch.png", show_shapes=True)

		predictions = np.argmax(model.predict(X_test), axis=2)
		print(predictions)

		sequences = []

		for i in predictions:
			test_sample_num += 1

			char_list = []
			for idx in i:
				if idx > 0:
					char_list.append(y_ix_to_word[idx])

			sequence = ''.join(char_list)
			print(test_sample_num,":", sequence)
			sequences.append(sequence)

		filename = "model2_out.txt"
		with open(filename, 'w', encoding='utf-8') as f:
			f.write("Words" + '\t' + 'Original Roots' + '\t' + "Predicted roots" + '\n')
			for a,b,c in zip(X_unique, y_unique, sequences):
				f.write(str(a) + '\t\t' + str(b) + '\t\t' + str(c) + '\n')

'''