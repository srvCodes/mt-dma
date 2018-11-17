import pickle
from load_data_with_phonetic import load_data_for_seq2seq, load_data_for_features

import keras.backend as K
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.layers import Multiply, Add, Lambda, Activation, TimeDistributed, Dense, RepeatVector, Embedding, Input, merge, \
	concatenate, GaussianNoise, dot 
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Layer
from keras.optimizers import Adam, Adadelta
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau, TensorBoard
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

# from curve_plotter import plot_precision_recall


EPOCHS = 500
dropout = 0.2
TIME_STEPS = 20
EMBEDDING_DIM = 64
BATCH_SIZE = 128
LAYER_NUM = 2
no_filters = 64
filter_length = 4
HIDDEN_DIM = no_filters*2
RNN = GRU
rnn_output_size = 32
folds = 10

class_labels = []


def removez_erroneous_indices(lists):
	to_be_removed = pickle.load(open('./pickle_dumps/removed_indices', 'rb'))
	to_be_removed = list(set(to_be_removed)) # for ascending order

	helper_cnt = 0
	for i in to_be_removed:
		i = i - helper_cnt
		for j in range(len(lists)):
			lists[j].pop(i)
		helper_cnt = helper_cnt + 1

	return lists


def write_words_to_file(orig_words, predictions):
	print("Writing to file ..")
	# print(sentences[:10])
	sentences = pickle.load(open('./pickle_dumps/test_words', 'rb'))

	X = [item for sublist in sentences for item in sublist]
	Y = [item for sublist in orig_words for item in sublist]

	# X, Y = remove_erroneous_indices([X,Y])

	filename = "./outputs/freezing_with_luong/multitask_context_out1.txt"
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

	sentences = pickle.load(open('./pickle_dumps/test_words', 'rb'))
	words = [item for sublist in sentences for item in sublist]

	for i in range(len(orig_features)):
		filename = "./outputs/freezing_with_luong/feature"+str(i)+"context_out1.txt"
		with open(filename, 'w', encoding='utf-8') as f:
			f.write("Word" + '\t\t' + 'Original feature' + '\t' + 'Predicted feature' + '\n')
			for a,b,c in zip(words, orig_features[i], pred_features[i]):
				f.write(str(a) + '\t\t' + str(b) + '\t\t\t' + str(c) + '\n')

	print("Success writing features to files !!")

def process_features(y1,y2,y3,y4,y5,y6, n = None, enc=None):

	y = [y1, y2, y3, y4, y5, y6]

	in_cnt1 = Counter(y1)
	in_cnt2 = Counter(y2)
	in_cnt3 = Counter(y3)
	in_cnt4 = Counter(y4)
	in_cnt5 = Counter(y5)
	in_cnt6 = Counter(y6)

	labels=[] # for processing of unnecessary labels from the test set
	init_cnt = [in_cnt1, in_cnt2, in_cnt3, in_cnt4, in_cnt5, in_cnt6]

	for i in range(len(init_cnt)):
		labels.append(list(init_cnt[i].keys()))

	if enc == None:
		enc = {}
		transformed = []
		print("processing train encoders!")
		for i in range(len(y)):
			enc[i] = LabelEncoder()
			enc[i].fit(y[i]+ ['unc'])
			transformed.append(enc[i].transform(y[i]))

	else:
		transformed = []
		print("processing test encoders !")
		for i in range(len(y)):
			arr = [w if w in list(enc[i].classes_) else 'unc' for w in y[i]]
			transformed.append(enc[i].transform(arr))

	y1 = list(transformed[0])
	y2 = list(transformed[1])
	y3 = list(transformed[2])
	y4 = list(transformed[3])
	y5 = list(transformed[4])
	y6 = list(transformed[5])

	cnt1 = Counter(y1)
	cnt2 = Counter(y2)
	cnt3 = Counter(y3)
	cnt4 = Counter(y4)
	cnt5 = Counter(y5)
	cnt6 = Counter(y6)	

	if enc != None:
		lis = [cnt1, cnt2, cnt3, cnt4, cnt5, cnt6]
		for i in range(len(lis)):
			class_labels.append(list(lis[i].keys()))

	if n == None:
		n1 = max(cnt1, key=int) + 1
		n2 = max(cnt2, key=int) + 1
		n3 = max(cnt3, key=int) + 1
		n4 = max(cnt4, key=int) + 1
		n5 = max(cnt5, key=int) + 1
		n6 = max(cnt6, key=int) + 1
	
	else:
		n1,n2,n3,n4,n5,n6 = n

	y1 = np_utils.to_categorical(y1, num_classes=n1)
	y2 = np_utils.to_categorical(y2, num_classes=n2)
	y3 = np_utils.to_categorical(y3, num_classes=n3)
	y4 = np_utils.to_categorical(y4, num_classes=n4)
	y5 = np_utils.to_categorical(y5, num_classes=n5)
	y6 = np_utils.to_categorical(y6, num_classes=n6)

	pickle.dump(enc, open('./pickle_dumps/enc','wb'))
	print("Encoders dumped ################################################")
	return (y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6, enc, labels)



def process_data(word_sentences, max_len, word_to_ix):
	# Vectorizing each element in each sequence
	sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
	for i, sentence in enumerate(word_sentences):
		for j, word in enumerate(sentence):
			sequences[i, j, word] = 1
	return sequences

def create_model(X_vocab_len, X_max_len, n_phonetic_features, y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6, hidden_size, num_layers):
	def smart_merge(vectors, **kwargs):
		return vectors[0] if len(vectors) == 1 else merge(vectors, **kwargs)

	current_word = Input(shape=(X_max_len,), dtype='float32', name='input1') # for encoder (shared)
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

	list_of_inputs = [current_word, right_word1, right_word2, right_word3,right_word4, 
					left_word1, left_word2, left_word3, left_word4]

	current_word_embedding, right_word_embedding1, right_word_embedding2,right_word_embedding3, right_word_embedding4, \
		left_word_embedding1, left_word_embedding2, left_word_embedding3, left_word_embedding4 = [emb_layer1(i) for i in list_of_inputs]

	print("Type:: ",type(current_word_embedding))
	list_of_embeddings1 = [current_word_embedding, right_word_embedding1, right_word_embedding2,right_word_embedding3, right_word_embedding4, \
		left_word_embedding1, left_word_embedding2, left_word_embedding3, left_word_embedding4]

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

	x = Dropout(0.15, name='drop_single1')(concat)
	x = Bidirectional(RNN(rnn_output_size, name='bidirec1'))(x)

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

	current_word_embedding, right_word_embedding1, right_word_embedding2,right_word_embedding3, right_word_embedding4, \
		left_word_embedding1, left_word_embedding2, left_word_embedding3, left_word_embedding4 = [emb_layer(i) for i in list_of_inputs]

	encoder_forward = GRU(rnn_output_size, return_sequences=True, unroll=True, kernel_regularizer=l2(0.0001), name='encoder_f')(current_word_embedding)
	encoder_forward_last = encoder_forward[:,-1,:]

	encoder_backward = GRU(rnn_output_size, return_sequences=True, unroll=True, go_backwards=True, kernel_regularizer=l2(0.0001), name='encoder_b')(current_word_embedding)
	encoder_backward_last = encoder_backward[:,-1,:]

	Bidirectional_enc = [encoder_forward, encoder_backward]
	encoder = smart_merge(Bidirectional_enc, mode='concat')

	decoder = emb_layer(decoder_input)
	decoder_forward = GRU(rnn_output_size, return_sequences=True, unroll=True, kernel_regularizer=l2(0.0001),name='decoder_f')(decoder, initial_state=[encoder_forward_last])
	decoder_backward = GRU(rnn_output_size, return_sequences=True, unroll=True, go_backwards=True, kernel_regularizer=l2(0.0001),name='decoder_b')(decoder, initial_state=[encoder_backward_last])

	Bidirectional_dec = [decoder_forward, decoder_backward]
	decoder = smart_merge(Bidirectional_dec, mode='concat')

	attention = dot([decoder, encoder], axes=[2,2], name='dot')
	attention = Activation('softmax', name='attention')(attention)

	context = dot([attention, encoder], axes=[2,1], name='dot2')
	decoder_combined_context = concatenate([context, decoder], name='concatenate')

	outputs = TimeDistributed(Dense(64, activation='tanh'), name='td1')(decoder_combined_context)
	outputs = TimeDistributed(Dense(X_vocab_len, activation='softmax'),  name='td2')(outputs)

	all_inputs = [current_word, decoder_input, right_word1, right_word2, right_word3, right_word4, left_word1, left_word2, left_word3,\
				  left_word4, phonetic_input]
	all_outputs = [outputs, out1, out2, out3, out4, out5, out6]

	model = Model(input=all_inputs, output=all_outputs)

	return model


sentences = pickle.load(open('./pickle_dumps/train_words', 'rb'))
rootwords = pickle.load(open('./pickle_dumps/train_roots', 'rb'))
features = pickle.load(open('./pickle_dumps/train_features', 'rb'))

n_phonetics, X_train_phonetics, X_test_phonetics, X_val_phonetics = returnTrainTestSets()

# we keep X_idx2word and y_idx2word the same
# X_left & X_right = X shifted to one and two positions left and right for context2
X, X_vocab_len, X_word_to_ix, X_ix_to_word, y, y_vocab_len, y_word_to_ix, y_ix_to_word, X_left1, X_left2, X_left3, X_left4, \
X_right1, X_right2, X_right3, X_right4 = load_data_for_seq2seq(sentences, rootwords, test=False, context4=True)

y1, y2, y3, y4, y5, y6 = load_data_for_features(features)

y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6, enc, labels = process_features(y1, y2, y3, y4, y5, y6)

n = [n1, n2, n3, n4, n5, n6]

X_max = max([len(word) for word in X])
y_max = max([len(word) for word in y])
X_max_len = max(X_max, y_max)

print(X_max_len)
exit(1)
pickle.dump(n, open('./pickle_dumps/n', 'wb'))
pickle.dump(X_max_len, open('./pickle_dumps/X-max-len', 'wb'))
pickle.dump(X_vocab_len, open('./pickle_dumps/X_vocab_len', 'wb'))

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
y = pad_sequences(y, maxlen = X_max_len, dtype = 'int32', padding='post')

print("Compiling Model ..")
model = create_model(X_vocab_len, X_max_len,  
					 n_phonetics, y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6, HIDDEN_DIM, LAYER_NUM)
model.compile(optimizer=Adadelta(), loss='categorical_crossentropy',
			  metrics=['accuracy'])

pseudo_model = create_model(X_vocab_len, X_max_len, n_phonetics,
					 y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6, HIDDEN_DIM, LAYER_NUM)

saved_weights = "./model_weights/frozen_training_weights.hdf5"

MODE = 'trai'

if MODE == 'train':
	print("Training model ..")
	plot_model(model, to_file="final_model.png", show_shapes=True)
	y_sequences = process_data(y, X_max_len, X_word_to_ix)

	print("X len ======== ", len(X))
	train_val_cutoff = int(.75 * len(X))
	X_train, X_left1_tr, X_left2_tr, X_left3_tr, X_left4_tr, X_right1_tr, X_right2_tr, X_right3_tr, X_right4_tr, y_train = \
		[X[:train_val_cutoff], X_left1[:train_val_cutoff], X_left2[:train_val_cutoff], X_left3[:train_val_cutoff], X_left4[:train_val_cutoff],
			X_right1[:train_val_cutoff], X_right2[:train_val_cutoff], X_right3[:train_val_cutoff], X_right4[:train_val_cutoff], y[:train_val_cutoff]]
	X_val, X_left1_val, X_left2_val, X_left3_val, X_left4_val, X_right1_val, X_right2_val, X_right3_val, X_right4_val, y_val = \
		[X[train_val_cutoff:], X_left1[train_val_cutoff:], X_left2[train_val_cutoff:], X_left3[train_val_cutoff:], X_left4[train_val_cutoff:],
			X_right1[train_val_cutoff:], X_right2[train_val_cutoff:], X_right3[train_val_cutoff:], X_right4[train_val_cutoff:], y[train_val_cutoff:]]

	y_sequences_tr, y1_tr, y2_tr, y3_tr, y4_tr, y5_tr, y6_tr = \
			[y_sequences[:train_val_cutoff], y1[:train_val_cutoff], y2[:train_val_cutoff], y3[:train_val_cutoff], \
				y4[:train_val_cutoff], y5[:train_val_cutoff], y6[:train_val_cutoff]]	
	y_sequences_val, y1_val, y2_val, y3_val, y4_val, y5_val, y6_val = \
			[y_sequences[train_val_cutoff:], y1[train_val_cutoff:], y2[train_val_cutoff:], y3[train_val_cutoff:], \
				y4[train_val_cutoff:], y5[train_val_cutoff:], y6[train_val_cutoff:]]

	X_decoder_input = np.zeros_like(X_train)
	X_decoder_input[:, 1:] = X_train[:,:-1]
	X_decoder_input[:, 0] = 1

	X_decoder_val = np.zeros_like(X_val)
	X_decoder_val[:, 1:] = X_val[:,:-1]
	X_decoder_val[:, 0] = 1

	# tb = TensorBoard('./initial_logs', write_graph=True, write_images=True, batch_size=BATCH_SIZE)
	# hist1 = model.fit([X_train, X_decoder_input,  X_right1_tr, X_right2_tr, X_right3_tr, X_right4_tr, X_left1_tr, X_left2_tr, X_left3_tr, X_left4_tr, X_train_phonetics],
	# 			 [y_sequences_tr, y1_tr, y2_tr, y3_tr, y4_tr, y5_tr, y6_tr],
	# 			 validation_data=([X_val, X_decoder_val,  X_right1_val, X_right2_val, X_right3_val, X_right4_val, X_left1_val, X_left2_val, X_left3_val, X_left4_val, X_val_phonetics],\
	# 			 [y_sequences_val, y1_val, y2_val, y3_val, y4_val, y5_val, y6_val]),
	# 			 batch_size=BATCH_SIZE, epochs=300, 
	# 			 callbacks=[EarlyStopping(patience=10),
	# 							ModelCheckpoint('./model_weights/training_weights.hdf5', save_best_only=True,
	# 											verbose=1, save_weights_only=True),
	# 							ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.01), tb])

	print("Setting weights now.. ")
	model.load_weights('./model_weights/training_weights.hdf5')
	names_to_be_freezed = ['Embedding', 'merge_1', 'drop1_0', 'drop1_1', 'drop1_2', 'drop1_3', 'drop1_4', 'drop1_5', 'drop1_6', 'drop1_7', 
		'drop1_8', 'noise1_0', 'noise1_1', 'noise1_2', 'noise1_3', 'noise1_4', 'noise1_5', 'noise1_6', 'noise1_7', 'noise1_8', 
		'conv4_0', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv4_5', 'conv4_6', 'conv4_7', 'conv4_8', 'conv5_0', 'conv5_1', 
		'conv5_2', 'conv5_3', 'conv5_4', 'conv5_5', 'conv5_6', 'conv5_7', 'conv5_8',  'max4_0', 'avg4_0', 'max4_1', 'avg4_1', 'max4_2', 
		'avg4_2', 'max4_3', 'avg4_3', 'max4_4', 'avg4_4', 'max4_5', 'avg4_5', 'max4_6', 'avg4_6', 'max4_7', 'avg4_7', 'max4_8', 'avg4_8', 
		'max5_0', 'avg5_0', 'max5_1', 'avg5_1', 'max5_2', 'avg5_2', 'max5_3', 'avg5_3', 'max5_4', 'avg5_4', 'max5_5', 'avg5_5', 'max5_6', 
		'avg5_6', 'max5_7', 'avg5_7', 'max5_8', 'avg5_8',  'merge_conv4_0', 'merge_conv4_1', 'merge_conv4_2', 'merge_conv4_3', 'merge_conv4_4',
		'merge_conv4_5', 'merge_conv4_6', 'merge_conv4_7', 'merge_conv4_8', 'merge_conv5_0', 'merge_conv5_1', 'merge_conv5_2', 'merge_conv5_3',
		'merge_conv5_4', 'merge_conv5_5', 'merge_conv5_6', 'merge_conv5_7', 'merge_conv5_8', 'merge_2', 'main_merge',  'drop_single1',
		'bidirec1', 'phonetic_merging',  'dense1', 'drop_single2', 'dense2',  'drop_single3', 'output1', 'output2', 'output3', 'output4', 
		'output5', 'output6']
	
	for (i,j) in zip(model.layers, pseudo_model.layers):
		if i.name == j.name:
			print("Setting weights of layer: " + str(i.name))
			pseudo_model.get_layer(j.name).set_weights(model.get_layer(i.name).get_weights())

	for name in names_to_be_freezed:
		print("Freezing layers now..")
		layer = pseudo_model.get_layer(name)
		if layer is not None:
			pseudo_model.get_layer(name).trainable = False

	pseudo_model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])
	tb2 = TensorBoard('./frozen_logs', write_graph=True, write_images=True, batch_size=BATCH_SIZE)
	hist = pseudo_model.fit([X_train, X_decoder_input,  X_right1_tr, X_right2_tr, X_right3_tr, X_right4_tr, X_left1_tr, X_left2_tr, X_left3_tr, X_left4_tr, X_train_phonetics],
				 [y_sequences_tr, y1_tr, y2_tr, y3_tr, y4_tr, y5_tr, y6_tr],
				 validation_data=([X_val, X_decoder_val,  X_right1_val, X_right2_val, X_right3_val, X_right4_val, X_left1_val, X_left2_val, X_left3_val, X_left4_val, X_val_phonetics],\
				 [y_sequences_val, y1_val, y2_val, y3_val, y4_val, y5_val, y6_val]),
				 batch_size=BATCH_SIZE, epochs=300,
				 callbacks=[EarlyStopping(patience=8),
				 			ModelCheckpoint('./model_weights/frozen_training_weights.hdf5',
												save_best_only=True, verbose=1, save_weights_only=True),
				 			ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.005), tb2])

	# print("Initial model performance !!")
	# plot_model_performance(
	# train_loss=hist1.history.get('loss', []),
 #    train_acc=hist1.history.get('acc', []),
 #    train_val_loss=hist1.history.get('val_loss', []),
 #    train_val_acc=hist1.history.get('val_acc', [])
	# )

	print("Frozen model performance !!")
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
		# FILENAME = './test_file.txt'
		test_sentences = pickle.load(open('./pickle_dumps/test_words', 'rb'))
		# test_sentences = [line for line in open(FILENAME)]
		# test_sentences = open(FILENAME).readlines()
		test_roots = pickle.load(open('./pickle_dumps/test_roots', 'rb'))
		test_features = pickle.load(open('./pickle_dumps/test_features', 'rb'))

		y1, y2, y3, y4, y5, y6 = load_data_for_features(test_features)
		features = [y1, y2, y3, y4, y5, y6]

		complete_list, X_test, X_vcab_len, X_wrd_to_ix, X_ix_to_wrd, y_test, y_vcab_len, y_wrd_to_ix, y_ix_to_wrd, \
		X_left1, X_left2, X_left3, X_left4, X_right1, X_right2, X_right3, X_right4, X_phonetic_features = \
			load_data_for_seq2seq(test_sentences, test_roots, X_test_phonetics, features, labels, test=True, context4=True)

		X_orig, y_orig, y1, y2, y3, y4, y5, y6 = complete_list

		to_be_padded = [X_test, X_left1, X_right1, X_left2, X_right2, X_left3, X_right3, X_left4, X_right4, y_test]

		X_test, X_left1, X_right1, X_left2, X_right2, X_left3, X_right3, X_left4, X_right4, y_test= \
						[pad_sequences(i, maxlen=X_max_len, dtype='int32', padding='post') for i in to_be_padded]

		y_test_seq = process_data(y_test, X_max_len, y_word_to_ix)

		y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6, enc, lab = process_features(y1, y2, y3, y4, y5, y6, 
																							n,
																							enc)  # pass previous encoders as args

		decoder_input = np.zeros_like(X_test)
		decoder_input[:, 1:] = X_test[:,:-1]
		decoder_input[:, 0] = 1

		
		model.load_weights(saved_weights)
		print(model.summary())

		# for i, layer in enumerate(model.layers):
		# 	print(i, layer.name)

		print(model.evaluate([X_test, decoder_input,  X_right1, X_right2, X_right3, X_right4, X_left1, X_left2, X_left3, X_left4, X_phonetic_features],
							 [y_test_seq, y1, y2, y3, y4, y5, y6]))
		# # # print(model.metrics_names)

		words, f1, f2, f3, f4, f5, f6 = model.predict(
			[X_test, decoder_input, X_right1, X_right2, X_right3, X_right4, X_left1, X_left2, X_left3, X_left4, X_phonetic_features])

		predictions = np.argmax(words, axis=2)
		pickle.dump(predictions, open('./pickle_dumps/predictions', 'wb'))

		pred_features = [f1, f2, f3, f4, f5, f6]
		orig_features = [y1, y2, y3, y4, y5, y6]

		# dump for generating graphs
		pickle.dump(pred_features, open('./pickle_dumps/predictions_rcnn_with_attention_and_roots', 'wb'))
		pickle.dump(orig_features, open('./pickle_dumps/originals', 'wb'))
		pickle.dump(n, open('./pickle_dumps/num_classes', 'wb'))
		pickle.dump(class_labels, open('./pickle_dumps/class_labels', 'wb'))

	
		# write to files
		f1, f2, f3, f4, f5, f6 = [np.argmax(i, axis=1) for i in pred_features]
		pred_features = [f1, f2, f3, f4, f5, f6]
		
		write_features_to_file(orig_features, pred_features, enc)

		# Post processing of predicted roots
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

		print(sequences[:10])
		write_words_to_file(test_roots, sequences)