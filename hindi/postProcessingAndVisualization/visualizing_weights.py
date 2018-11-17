import pickle
from load_data_with_phonetic import load_data_for_seq2seq, load_data_for_features

import keras.backend as K
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Multiply, Add, Lambda, Activation, TimeDistributed, Dense, RepeatVector, Embedding, Input, merge, \
	concatenate, GaussianNoise, dot 
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Layer
from keras.optimizers import Adam, Adadelta
from keras.utils import plot_model, np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import initializers, regularizers, constraints
from keras.layers import Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, BatchNormalization
from keras.regularizers import l2
from keras.constraints import maxnorm

from nltk import FreqDist
import numpy as np
import sys, time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter, deque
from predict_with_features import plot_model_performance, returnTrainTestSets
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

MODE = 'trai'
output_mode = 'dump'

EPOCHS = 500
dropout = 0.2
TIME_STEPS = 20
EMBEDDING_DIM = 128
BATCH_SIZE = 128
LAYER_NUM = 2
no_filters = 128
filter_length = 4
HIDDEN_DIM = no_filters*2
RNN = GRU
rnn_output_size = 32
folds = 10

class_labels = []

def dump_processed_inputs(item, name):
	src = './processed_inputs/'
	pickle.dump(item, open(src+name, 'wb'))


def write_words_to_file(orig_words, predictions):
	print("Writing to file ..")
	# print(sentences[:10])

	sentences = pickle.load(open('./pickle-dumps/sentences_test', 'rb'))

	X = [item for sublist in sentences for item in sublist]
	Y = [item for sublist in orig_words for item in sublist]

	filename = "./outputs/CNNRNN_luong_attention_with_roots/multitask_context_out.txt"
	with open(filename, 'w', encoding='utf-8') as f:
		f.write("Words" + '\t\t\t' + 'Original Roots' + '\t\t' + "Predicted roots" + '\n')
		for a, b, c in zip(X, Y, predictions):
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
		filename = "./outputs/CNNRNN_luong_attention_with_roots/feature"+str(i)+"context_out.txt"
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



def process_data(word_sentences, max_len, word_to_ix):
	# Vectorizing each element in each sequence
	sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
	for i, sentence in enumerate(word_sentences):
		for j, word in enumerate(sentence):
			sequences[i, j, word] = 1
	return sequences


def get_truncated_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, n_phonetic_features, y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6,
				 hidden_size, num_layers):
	def smart_merge(vectors, **kwargs):
		return vectors[0] if len(vectors) == 1 else merge(vectors, **kwargs)

	current_word = Input(shape=(X_max_len,), dtype='float32', name='input1') # for encoder (shared)
	root_word = Input(shape=(X_max_len,), dtype='float32', name='input2')
	decoder_input = Input(shape=(X_max_len,), dtype='float32', name='input3') # for decoder -- attention
	right_word1 = Input(shape=(X_max_len,), dtype='float32', name='input4')
	right_word2 = Input(shape=(X_max_len,), dtype='float32', name='input5')
	right_word3 = Input(shape=(X_max_len,), dtype='float32', name='input6')
	right_word4 = Input(shape=(X_max_len,), dtype='float32', name='input7')
	left_word1 = Input(shape=(X_max_len,), dtype='float32', name='input8')
	left_word2 = Input(shape=(X_max_len,), dtype='float32', name='input9')
	left_word3 = Input(shape=(X_max_len,), dtype='float32', name='input10')
	left_word4 = Input(shape=(X_max_len,), dtype='float32', name='input11')
	phonetic_input = Input(shape=(n_phonetic_features,), dtype='float32', name='input12')

	emb_layer1 = Embedding(X_vocab_len, EMBEDDING_DIM,
						  input_length=X_max_len,
						  mask_zero=False, name='Embedding')

	list_of_inputs = [current_word, root_word, right_word1, right_word2, right_word3,right_word4, 
					left_word1, left_word2, left_word3, left_word4]

	current_word_embedding, root_word_embedding, right_word_embedding1, right_word_embedding2,right_word_embedding3, right_word_embedding4, \
		left_word_embedding1, left_word_embedding2, left_word_embedding3, left_word_embedding4 = [emb_layer1(i) for i in list_of_inputs]

	print("Typeeeee:: ",type(current_word_embedding))
	current_word_embedding = smart_merge([current_word_embedding, root_word_embedding]) # concatenate root word with current input
	list_of_embeddings1 = [current_word_embedding, right_word_embedding1, right_word_embedding2,right_word_embedding3, right_word_embedding4, \
		left_word_embedding1, left_word_embedding2, left_word_embedding3, left_word_embedding4]

	# list_of_embeddings = [smart_merge([i,root_word_embedding]) for i in list_of_embeddings] # concatenate root word with each of inputs
	list_of_embeddings = [Dropout(0.50, name='drop1_'+str(j))(i) for i,j in zip(list_of_embeddings1, range(len(list_of_embeddings1)))]
	list_of_embeddings = [GaussianNoise(0.05, name='noise1_'+str(j))(i) for i,j in zip(list_of_embeddings, range(len(list_of_embeddings)))]
	
	conv4_curr, conv4_right1, conv4_right2, conv4_right3, conv4_right4, conv4_left1, conv4_left2, conv4_left3, conv4_left4 =\
			[Conv1D(filters=no_filters, 
				kernel_size=4, padding='valid',activation='relu', 
				strides=1, name='conv4_'+str(j))(i) for i,j in zip(list_of_embeddings, range(len(list_of_embeddings)))]

	conv4s = [conv4_curr, conv4_right1, conv4_right2, conv4_right3, conv4_right4, conv4_left1, conv4_left2, conv4_left3, conv4_left4]
	maxPool4 = [MaxPooling1D(name='max4_'+str(j))(i) for i,j in zip(conv4s, range(len(conv4s)))]
	avgPool4 = [AveragePooling1D(name='avg4_'+str(j))(i) for i,j in zip(conv4s, range(len(conv4s)))]

	pool4_curr, pool4_right1, pool4_right2, pool4_right3, pool4_right4, pool4_left1, pool4_left2, pool4_left3, pool4_left4 = \
		[merge([i,j], name='merge_conv4_'+str(k)) for i,j,k in zip(maxPool4, avgPool4, range(len(maxPool4)))]

	conv5_curr, conv5_right1, conv5_right2, conv5_right3, conv5_right4, conv5_left1, conv5_left2, conv5_left3, conv5_left4 = \
			[Conv1D(filters=no_filters,
				kernel_size=5,
				padding='valid',
				activation='relu',
				strides=1, name='conv5_'+str(j))(i) for i,j in zip(list_of_embeddings, range(len(list_of_embeddings)))]	

	conv5s = [conv5_curr, conv5_right1, conv5_right2, conv5_right3, conv5_right4, conv5_left1, conv5_left2, conv5_left3, conv5_left4]
	maxPool5 = [MaxPooling1D(name='max5_'+str(j))(i) for i,j in zip(conv5s, range(len(conv5s)))]
	avgPool5 = [AveragePooling1D(name='avg5_'+str(j))(i) for i,j in zip(conv5s, range(len(conv5s)))]

	pool5_curr, pool5_right1, pool5_right2, pool5_right3, pool5_right4, pool5_left1, pool5_left2, pool5_left3, pool5_left4 = \
		[merge([i,j], name='merge_conv5_'+str(k)) for i,j,k in zip(maxPool5, avgPool5, range(len(maxPool5)))]


	maxPools = [pool4_curr, pool4_right1, pool4_right2, pool4_right3, pool4_right4, \
		pool4_left1, pool4_left2, pool4_left3, pool4_left4, \
		pool5_curr, pool5_right1, pool5_right2, pool5_right3, pool5_right4, \
		pool5_left1, pool5_left2, pool5_left3, pool5_left4]

	concat = merge(maxPools, mode='concat', name='main_merge')
	# curr_vector_total = smart_merge([pool4_curr, pool5_curr], mode='concat')

	x = Dropout(0.15, name='drop_single1')(concat)

	x = Bidirectional(RNN(rnn_output_size, name='rnn_for_features'))(x)

	total_features = [x, phonetic_input]
	concat2 = merge(total_features, mode='concat', name='phonetic_merging')

	x = Dense(HIDDEN_DIM, activation='relu', kernel_initializer='he_normal',
			  kernel_constraint= maxnorm(3), bias_constraint=maxnorm(3), name='dense1')(concat2)
	x = Dropout(0.15, name='drop_single2')(x)

	x = Dense(HIDDEN_DIM, kernel_initializer='he_normal', activation='tanh',
			  kernel_constraint= maxnorm(3), bias_constraint=maxnorm(3), name='dense2')(x)
	x = Dropout(0.15, name='drop_single3')(x)

	# out1 = Dense(n1, kernel_initializer='he_normal', activation='softmax', name='output1')(x)
	# out2 = Dense(n2, kernel_initializer='he_normal', activation='softmax', name='output2')(x)
	# out3 = Dense(n3, kernel_initializer='he_normal', activation='softmax', name='output3')(x)
	# out4 = Dense(n4, kernel_initializer='he_normal', activation='softmax', name='output4')(x)
	# out5 = Dense(n5, kernel_initializer='he_normal', activation='softmax', name='output5')(x)
	# out6 = Dense(n6, kernel_initializer='he_normal', activation='softmax', name='output6')(x)
			
	# # Luong et al. 2015 attention model	
	# emb_layer = Embedding(X_vocab_len, EMBEDDING_DIM,
	# 					  input_length=X_max_len,
	# 					  mask_zero=True, name='Embedding_for_seq2seq')

	# current_word_embedding = emb_layer(current_word)
	# current_word_embedding = GaussianNoise(0.05, name='noise_seq2seq')(current_word_embedding)

	# encoder, state = RNN(rnn_output_size, return_sequences=True, unroll=True, return_state=True, name='encoder')(current_word_embedding)
	# encoder_last = encoder[:,-1,:]

	# decoder = emb_layer(decoder_input)
	# decoder = GRU(rnn_output_size, return_sequences=True, unroll=True, name='decoder')(decoder, initial_state=[state])

	# attention = dot([decoder, encoder], axes=[2,2], name='dot')
	# attention = Activation('softmax', name='attention')(attention)

	# context = dot([attention, encoder], axes=[2,1], name='dot2')
	# decoder_combined_context = concatenate([context, decoder], name='concatenate')

	# outputs = TimeDistributed(Dense(64, activation='tanh', name='TimeDistributed1'))(decoder_combined_context)
	# outputs = TimeDistributed(Dense(X_vocab_len, activation='softmax', name='TimeDistributed2'))(outputs)

	all_inputs = [current_word, root_word, decoder_input, right_word1, right_word2, right_word3, right_word4, left_word1, \
				left_word2, left_word3, left_word4, phonetic_input]
	all_outputs = [x]

	model = Model(input=all_inputs, output=all_outputs)
	return model

def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, n_phonetic_features, y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6,
				 hidden_size, num_layers):
	def smart_merge(vectors, **kwargs):
		return vectors[0] if len(vectors) == 1 else merge(vectors, **kwargs)

	current_word = Input(shape=(X_max_len,), dtype='float32', name='input1') # for encoder (shared)
	root_word = Input(shape=(X_max_len,), dtype='float32', name='input2')
	decoder_input = Input(shape=(X_max_len,), dtype='float32', name='input3') # for decoder -- attention
	right_word1 = Input(shape=(X_max_len,), dtype='float32', name='input4')
	right_word2 = Input(shape=(X_max_len,), dtype='float32', name='input5')
	right_word3 = Input(shape=(X_max_len,), dtype='float32', name='input6')
	right_word4 = Input(shape=(X_max_len,), dtype='float32', name='input7')
	left_word1 = Input(shape=(X_max_len,), dtype='float32', name='input8')
	left_word2 = Input(shape=(X_max_len,), dtype='float32', name='input9')
	left_word3 = Input(shape=(X_max_len,), dtype='float32', name='input10')
	left_word4 = Input(shape=(X_max_len,), dtype='float32', name='input11')
	phonetic_input = Input(shape=(n_phonetic_features,), dtype='float32', name='input12')

	emb_layer1 = Embedding(X_vocab_len, EMBEDDING_DIM,
						  input_length=X_max_len,
						  mask_zero=False, name='Embedding')

	list_of_inputs = [current_word, root_word, right_word1, right_word2, right_word3,right_word4, 
					left_word1, left_word2, left_word3, left_word4]

	current_word_embedding, root_word_embedding, right_word_embedding1, right_word_embedding2,right_word_embedding3, right_word_embedding4, \
		left_word_embedding1, left_word_embedding2, left_word_embedding3, left_word_embedding4 = [emb_layer1(i) for i in list_of_inputs]

	print("Typeeeee:: ",type(current_word_embedding))
	current_word_embedding = smart_merge([current_word_embedding, root_word_embedding]) # concatenate root word with current input
	list_of_embeddings1 = [current_word_embedding, right_word_embedding1, right_word_embedding2,right_word_embedding3, right_word_embedding4, \
		left_word_embedding1, left_word_embedding2, left_word_embedding3, left_word_embedding4]

	# list_of_embeddings = [smart_merge([i,root_word_embedding]) for i in list_of_embeddings] # concatenate root word with each of inputs
	list_of_embeddings = [Dropout(0.50, name='drop1_'+str(j))(i) for i,j in zip(list_of_embeddings1, range(len(list_of_embeddings1)))]
	list_of_embeddings = [GaussianNoise(0.05, name='noise1_'+str(j))(i) for i,j in zip(list_of_embeddings, range(len(list_of_embeddings)))]
	
	conv4_curr, conv4_right1, conv4_right2, conv4_right3, conv4_right4, conv4_left1, conv4_left2, conv4_left3, conv4_left4 =\
			[Conv1D(filters=no_filters, 
				kernel_size=4, padding='valid',activation='relu', 
				strides=1, name='conv4_'+str(j))(i) for i,j in zip(list_of_embeddings, range(len(list_of_embeddings)))]

	conv4s = [conv4_curr, conv4_right1, conv4_right2, conv4_right3, conv4_right4, conv4_left1, conv4_left2, conv4_left3, conv4_left4]
	maxPool4 = [MaxPooling1D(name='max4_'+str(j))(i) for i,j in zip(conv4s, range(len(conv4s)))]
	avgPool4 = [AveragePooling1D(name='avg4_'+str(j))(i) for i,j in zip(conv4s, range(len(conv4s)))]

	pool4_curr, pool4_right1, pool4_right2, pool4_right3, pool4_right4, pool4_left1, pool4_left2, pool4_left3, pool4_left4 = \
		[merge([i,j], name='merge_conv4_'+str(k)) for i,j,k in zip(maxPool4, avgPool4, range(len(maxPool4)))]

	conv5_curr, conv5_right1, conv5_right2, conv5_right3, conv5_right4, conv5_left1, conv5_left2, conv5_left3, conv5_left4 = \
			[Conv1D(filters=no_filters,
				kernel_size=5,
				padding='valid',
				activation='relu',
				strides=1, name='conv5_'+str(j))(i) for i,j in zip(list_of_embeddings, range(len(list_of_embeddings)))]	

	conv5s = [conv5_curr, conv5_right1, conv5_right2, conv5_right3, conv5_right4, conv5_left1, conv5_left2, conv5_left3, conv5_left4]
	maxPool5 = [MaxPooling1D(name='max5_'+str(j))(i) for i,j in zip(conv5s, range(len(conv5s)))]
	avgPool5 = [AveragePooling1D(name='avg5_'+str(j))(i) for i,j in zip(conv5s, range(len(conv5s)))]

	pool5_curr, pool5_right1, pool5_right2, pool5_right3, pool5_right4, pool5_left1, pool5_left2, pool5_left3, pool5_left4 = \
		[merge([i,j], name='merge_conv5_'+str(k)) for i,j,k in zip(maxPool5, avgPool5, range(len(maxPool5)))]


	maxPools = [pool4_curr, pool4_right1, pool4_right2, pool4_right3, pool4_right4, \
		pool4_left1, pool4_left2, pool4_left3, pool4_left4, \
		pool5_curr, pool5_right1, pool5_right2, pool5_right3, pool5_right4, \
		pool5_left1, pool5_left2, pool5_left3, pool5_left4]

	concat = merge(maxPools, mode='concat', name='main_merge')
	# curr_vector_total = smart_merge([pool4_curr, pool5_curr], mode='concat')

	x = Dropout(0.15, name='drop_single1')(concat)

	x = Bidirectional(RNN(rnn_output_size, name='rnn_for_features'))(x)

	total_features = [x, phonetic_input]
	concat2 = merge(total_features, mode='concat', name='phonetic_merging')

	x = Dense(HIDDEN_DIM, activation='relu', kernel_initializer='he_normal',
			  kernel_constraint= maxnorm(3), bias_constraint=maxnorm(3), name='dense1')(concat2)
	x = Dropout(0.15, name='drop_single2')(x)

	x = Dense(HIDDEN_DIM, kernel_initializer='he_normal', activation='tanh',
			  kernel_constraint= maxnorm(3), bias_constraint=maxnorm(3), name='dense2')(x)
	x = Dropout(0.15, name='drop_single3')(x)

	out1 = Dense(n1, kernel_initializer='he_normal', activation='softmax', name='output1')(x)
	out2 = Dense(n2, kernel_initializer='he_normal', activation='softmax', name='output2')(x)
	out3 = Dense(n3, kernel_initializer='he_normal', activation='softmax', name='output3')(x)
	out4 = Dense(n4, kernel_initializer='he_normal', activation='softmax', name='output4')(x)
	out5 = Dense(n5, kernel_initializer='he_normal', activation='softmax', name='output5')(x)
	out6 = Dense(n6, kernel_initializer='he_normal', activation='softmax', name='output6')(x)
			
	# Luong et al. 2015 attention model	
	emb_layer = Embedding(X_vocab_len, EMBEDDING_DIM,
						  input_length=X_max_len,
						  mask_zero=True, name='Embedding_for_seq2seq')

	current_word_embedding = emb_layer(current_word)
	current_word_embedding = GaussianNoise(0.05, name='noise_seq2seq')(current_word_embedding)

	encoder, state = RNN(rnn_output_size, return_sequences=True, unroll=True, return_state=True, name='encoder')(current_word_embedding)
	encoder_last = encoder[:,-1,:]

	decoder = emb_layer(decoder_input)
	decoder = GRU(rnn_output_size, return_sequences=True, unroll=True, name='decoder')(decoder, initial_state=[state])

	attention = dot([decoder, encoder], axes=[2,2], name='dot')
	attention = Activation('softmax', name='attention')(attention)

	context = dot([attention, encoder], axes=[2,1], name='dot2')
	decoder_combined_context = concatenate([context, decoder], name='concatenate')

	outputs = TimeDistributed(Dense(64, activation='tanh', name='TimeDistributed1'))(decoder_combined_context)
	outputs = TimeDistributed(Dense(X_vocab_len, activation='softmax', name='TimeDistributed2'))(outputs)


	all_inputs = [current_word, root_word, decoder_input, right_word1, right_word2, right_word3, right_word4, left_word1, \
				left_word2, left_word3, left_word4, phonetic_input]
	all_outputs = [outputs, out1, out2, out3, out4, out5, out6]

	model = Model(input=all_inputs, output=all_outputs)
	opt = Adam()
	model.compile(optimizer=Adadelta(epsilon=1e-06), loss='categorical_crossentropy',
				  metrics=['accuracy'],
				  loss_weights=[1., 1., 1., 1., 1., 1., 1.])

	return model

sentences = pickle.load(open('./pickle-dumps/sentences_intra', 'rb'))
rootwords = pickle.load(open('./pickle-dumps/rootwords_intra', 'rb'))
features = pickle.load(open('./pickle-dumps/features_intra', 'rb'))

# n_phonetics, X_train_phonetics, X_test_phonetics, X_val_phonetics, _, _, _ = returnTrainTestSets()

# X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word, X_left1, X_left2, X_left3, X_left4, \
# X_right1, X_right2, X_right3, X_right4 = load_data_for_seq2seq(sentences, rootwords, test=False, context4=True)

# y1, y2, y3, y4, y5, y6, y7, y8 = load_data_for_features(features)

# y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n7, y8, n8, enc, labels = process_features(y1, y2, y3, y4, y5, y7, y8)

# n = [n1, n2, n3, n4, n5, n7, n8]

# X_max = max([len(word) for word in X])
# y_max = max([len(word) for word in y])
# X_max_len = max(X_max, y_max)

# print("Zero padding .. ")
# X = pad_sequences(X, maxlen= X_max_len, dtype = 'int32', padding='post')
# X_left1 = pad_sequences(X_left1, maxlen = X_max_len, dtype='int32', padding='post')
# X_left2 = pad_sequences(X_left2, maxlen = X_max_len, dtype='int32', padding='post')
# X_left3 = pad_sequences(X_left3, maxlen = X_max_len, dtype='int32', padding='post')
# X_left4 = pad_sequences(X_left4, maxlen = X_max_len, dtype='int32', padding='post')
# X_right1 = pad_sequences(X_right1, maxlen = X_max_len, dtype='int32', padding='post')
# X_right2 = pad_sequences(X_right2, maxlen = X_max_len, dtype='int32', padding='post')
# X_right3 = pad_sequences(X_right3, maxlen = X_max_len, dtype='int32', padding='post')
# X_right4 = pad_sequences(X_right4, maxlen = X_max_len, dtype='int32', padding='post')
# y = pad_sequences(y, maxlen = X_max_len, dtype = 'int32', padding='post')

# dump_processed_inputs(X, 'X')
# dump_processed_inputs(X_left1, 'X_left1')
# dump_processed_inputs(X_left2, 'X_left2')
# dump_processed_inputs(X_left3, 'X_left3')
# dump_processed_inputs(X_left4, 'X_left4')
# dump_processed_inputs(X_right1, 'X_right1')
# dump_processed_inputs(X_right4, 'X_right4')
# dump_processed_inputs(X_right3, 'X_right3')
# dump_processed_inputs(X_right2, 'X_right2')
# dump_processed_inputs(y, 'y')
# dump_processed_inputs(y1, 'y1')
# dump_processed_inputs(y2, 'y2')
# dump_processed_inputs(y3, 'y3')
# dump_processed_inputs(y4, 'y4')
# dump_processed_inputs(y5, 'y5')
# dump_processed_inputs(y7, 'y7')
# dump_processed_inputs(y8, 'y8')
# dump_processed_inputs(n1, 'n1')
# dump_processed_inputs(n2, 'n2')
# dump_processed_inputs(n3, 'n3')
# dump_processed_inputs(n4, 'n4')
# dump_processed_inputs(n5, 'n5')
# dump_processed_inputs(n7, 'n7')
# dump_processed_inputs(n8, 'n8')
# dump_processed_inputs(X_vocab_len, 'X_vocab_len')
# dump_processed_inputs(X_max_len, 'X_max_len')
# dump_processed_inputs(y_vocab_len, 'y_vocab_len')
# dump_processed_inputs(n_phonetics, 'n_phonetics')
# dump_processed_inputs(X_train_phonetics, 'X_train_phonetics')
# dump_processed_inputs(X_test_phonetics, 'X_test_phonetics')
# dump_processed_inputs(X_val_phonetics, 'X_val_phonetics')
# dump_processed_inputs(labels, 'labels')
# dump_processed_inputs(y_word_to_ix, 'y_word_to_ix')
# dump_processed_inputs(enc, 'enc')

src = './processed_inputs/'
X = pickle.load(open(src+'X', 'rb'))
X_left1 = pickle.load(open(src+'X_left1', 'rb'))
X_left2 = pickle.load(open(src+'X_left2', 'rb'))
X_left3 = pickle.load(open(src+'X_left3', 'rb'))
X_left4 = pickle.load(open(src+'X_left4', 'rb'))
X_right1 = pickle.load(open(src+'X_right1', 'rb'))
X_right2 = pickle.load(open(src+'X_right2', 'rb'))
X_right3 = pickle.load(open(src+'X_right3', 'rb'))
X_right4 = pickle.load(open(src+'X_right4', 'rb'))
y = pickle.load(open(src+'y', 'rb'))
y1 = pickle.load(open(src+'y1', 'rb'))
y2 = pickle.load(open(src+'y2', 'rb'))
y3 = pickle.load(open(src+'y3', 'rb'))
y4 = pickle.load(open(src+'y4', 'rb'))
y5 = pickle.load(open(src+'y5', 'rb'))
y7 = pickle.load(open(src+'y7', 'rb'))
y8 = pickle.load(open(src+'y8', 'rb'))
n1 = pickle.load(open(src+'n1', 'rb'))
n2 = pickle.load(open(src+'n2', 'rb'))
n3 = pickle.load(open(src+'n3', 'rb'))
n5 = pickle.load(open(src+'n5', 'rb'))
n4 = pickle.load(open(src+'n4', 'rb'))
n5 = pickle.load(open(src+'n5', 'rb'))
n7 = pickle.load(open(src+'n7', 'rb'))
n8 = pickle.load(open(src+'n8', 'rb'))
X_vocab_len = pickle.load(open(src+'X_vocab_len', 'rb'))
X_max_len = pickle.load(open(src+'X_max_len', 'rb'))
y_vocab_len = pickle.load(open(src+'y_vocab_len', 'rb'))
n_phonetics = pickle.load(open(src+'n_phonetics', 'rb'))
labels = pickle.load(open(src+'labels', 'rb'))
y_word_to_ix = pickle.load(open(src+'y_word_to_ix', 'rb'))
enc = pickle.load(open(src+'enc', 'rb'))
X_train_phonetics = pickle.load(open(src+'X_train_phonetics', 'rb'))
X_val_phonetics = pickle.load(open(src+'X_val_phonetics', 'rb'))
X_test_phonetics = pickle.load(open(src+'X_test_phonetics', 'rb'))
n = [n1, n2, n3, n4, n5, n7, n8]

print("Compiling Model ..")
trained_model = create_model(X_vocab_len, X_max_len, y_vocab_len, X_max_len, n_phonetics,
					y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n7, HIDDEN_DIM, LAYER_NUM)

truncated_model = get_truncated_model(X_vocab_len, X_max_len, y_vocab_len, X_max_len, n_phonetics,
					y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n7, HIDDEN_DIM, LAYER_NUM)

saved_weights = "./model_weights/charCNN_with_attention.hdf5"
whole_model = "./model_weights/attention_with_roots.hdf5"

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

	complete_list, X_test, X_vcab_len, X_wrd_to_ix, X_ix_to_wrd, y_test, y_vcab_len, y_wrd_to_ix, y_ix_to_wrd, \
	X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4, X_phonetic_features = \
		load_data_for_seq2seq(test_sentences, test_roots, X_test_phonetics, features, labels, test=True, context4=True)


	X_orig, y_orig, y1, y2, y3, y4, y5, y7, y8 = complete_list

	to_be_padded = [X_test, X_left1, X_right1, X_left2, X_right2, X_left3, X_right3, X_left4, X_right4, y_test]

	X_test, X_left1, X_right1, X_left2, X_right2, X_left3, X_right3, X_left4, X_right4, y_test= \
					[pad_sequences(i, maxlen=X_max_len, dtype='int32', padding='post') for i in to_be_padded]

	y_test_seq = process_data(y_test, X_max_len, y_word_to_ix)

	y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y7, n7, y8, n8, enc, lab = process_features(y1, y2, y3, y4, y5, y7, y8,
																						n,
																						enc)  # pass previous encoders as args

	decoder_input = np.zeros_like(X_test)
	decoder_input[:, 1:] = X_test[:,:-1]
	decoder_input[:, 0] = 1

	# trained_model = load_model(whole_model)
	trained_model.load_weights(saved_weights)
	print(trained_model.summary())

	names = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8', 'input9', 'input10', 'input11', 'input12' 
	'drop1_0', 'drop1_1', 'drop1_2', 'drop1_3', 'drop1_4', 'drop1_5', 'drop1_6', 'drop1_7', 'drop1_8',
	'noise1_0', 'noise1_1', 'noise1_2', 'noise1_3', 'noise1_4', 'noise1_5', 'noise1_6', 'noise1_7', 'noise1_8',
	'conv4_0', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv4_5', 'conv4_6', 'conv4_7', 'conv4_8',
	'conv5_0', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'conv5_5', 'conv5_6', 'conv5_7', 'conv5_8',
	'max4_1', 'max4_2', 'max4_3', 'max4_4', 'max4_5', 'max4_6', 'max4_7', 'max4_8', 'max4_0', 
	'avg4_0', 'avg4_1', 'avg4_2', 'avg4_3', 'avg4_4', 'avg4_5', 'avg4_6', 'avg4_7', 'avg4_8',
	'max5_1', 'max5_2', 'max5_3', 'max5_4', 'max5_5', 'max5_6', 'max5_7', 'max5_8', 'max5_0', 
	'avg5_0', 'avg5_1', 'avg5_2', 'avg5_3', 'avg5_4', 'avg5_5', 'avg5_6', 'avg5_7', 'avg5_8', 
	'merge_conv4_0', 'merge_conv4_1', 'merge_conv4_2', 'merge_conv4_3', 'merge_conv4_4', 'merge_conv4_5', 'merge_conv4_6', 'merge_conv4_7', 'merge_conv4_8', 
	'merge_conv5_0', 'merge_conv5_1', 'merge_conv5_2', 'merge_conv5_3', 'merge_conv5_4', 'merge_conv5_5', 'merge_conv5_6', 'merge_conv5_7', 'merge_conv5_8', 
	'noise_seq2seq', 'main_merge', 'encoder', 'drop_single1', 'decoder', 'rnn_for_features', 'dot', 'phonetic_merging', 'attention',
	'dense1', 'dot2', 'drop_single2', 'concatenate', 'dense2', 'drop_single3', 'output1', 'output2']

	for i in names:
		print(i)
		original_layer = trained_model.get_layer(i)
		truncated_model_layer = truncated_model.get_layer(i)
		if truncated_model_layer is None:
			continue
		
		truncated_model.get_layer(i).set_weights(original_layer.get_weights())
		
	truncated_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

	hidden_features = truncated_model.predict([X_test, y_test, decoder_input, X_left1, X_left2, 
		X_left3, X_left4, X_right1, X_right2, X_right3, X_right4, X_phonetic_features])

	pca = PCA(n_components = 20)
	pca_result = pca.fit_transform(hidden_features)
	print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
	##Variance PCA: 0.993621154832802

	src = './pca_outputs_for_tsne/'
	pickle.dump(pca_result, open(src+'pca_result', 'wb'))
	pickle.dump(y1, open(src+'y1', 'wb'))
	pickle.dump(y2, open(src+'y2', 'wb'))
	pickle.dump(y3, open(src+'y3', 'wb'))
	pickle.dump(y4, open(src+'y4', 'wb'))
	pickle.dump(y5, open(src+'y5', 'wb'))
	pickle.dump(y7, open(src+'y7', 'wb'))
	pickle.dump(n1, open(src+'n1', 'wb'))
	pickle.dump(n2, open(src+'n2', 'wb'))
	pickle.dump(n3, open(src+'n3', 'wb'))
	pickle.dump(n4, open(src+'n4', 'wb'))
	pickle.dump(n5, open(src+'n5', 'wb'))
	pickle.dump(n7, open(src+'n7', 'wb'))

	#Run T-SNE on the PCA features.
	tsne = TSNE(n_components=n5,  verbose = 1, method='exact')
	tsne_results = tsne.fit_transform(pca_result[:15000])

	y_test_cat = y5[:15000]
	color_map = np.argmax(y_test_cat, axis=1)
	plt.figure(figsize=(10,10))
	for cl in range(n5):
	    indices = np.where(color_map==cl)
	    indices = indices[0]
	    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl)
	plt.legend()
	plt.show()

