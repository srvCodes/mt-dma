import os
import re
import pickle 
from collections import Counter

path = "/home/saurav/Documents/urdu_morph_analysis/UD_Urdu-UDTB-master/ur_udtb-ud-dev.conllu"


def findString(string, pattern):
	try:
		start = string.index(pattern) + len(pattern)
		end = string.find('|', string.index(pattern))
		return string[start:end]
	except ValueError:
		return "Unk" # in case the attribute is absent

def get_useful_features(total_features):
	pos = total_features[0]
	features = '|'.join(total_features[2:])
	case = findString(features, 'Case=')
	gen = findString(features, 'Gender=')
	num = findString(features, 'Number=')
	person = findString(features, 'Person=')
	tam = findString(features, 'Tam=')

	return [pos, gen, num, person, case, tam]

def get_everything(lines):
	lines = lines[2:]

	words = []
	roots = []
	features = []

	for line in lines:
		lis = re.split(r'\t+', line.rstrip('\t'))
		words.insert(len(words), lis[1])
		roots.insert(len(words), lis[2])
		total_features = lis[3:]
		
		features.insert(len(features), get_useful_features(total_features))

	return words, roots, features

def showSentenceWiseFeatures():
	sentences = pickle.load(open(dump_path+"train_words", 'rb'))
	print("Total sentences:", len(sentences))

	# mean sentence len
	slen = 0
	for s in sentences:
		#print(s)
		slen += len(s)
	print("Mean len: ", slen/len(sentences))

	# no of unique words
	all_words = [item for sentence in sentences for item in sentence]
	print("Total words: ", len(all_words))
	words_set = set(all_words)
	print("Unique words: ", len(words_set))

def showTypesOfFeatures():
	features = pickle.load(open(dump_path+"train_features", 'rb'))

	pos = []; gender = []; num = []; person = []; case = []; tam = []
	for sentence in features:
		pos += [word[0] for word in sentence]
		gender += [word[1] for word in sentence]
		num += [word[2] for word in sentence]
		person += [word[3] for word in sentence]
		case += [word[4] for word in sentence]
		tam += [word[5] for word in sentence]

	# print(set(gender))
	print("cat: ", Counter(pos))
	print("gender: ", Counter(gender))
	print("number: ", Counter(num))
	print("person: ", Counter(person))
	print("Case: ", Counter(case))
	print("TAM: ", Counter(tam))

if __name__ == "__main__":
	dump_path = "./pickle_dumps/"
	# words = []
	# roots = []
	# features = []
	
	words = pickle.load(open(dump_path+'train_words', 'rb'))
	roots = pickle.load(open(dump_path+'train_roots', 'rb'))
	features = pickle.load(open(dump_path+'train_features', 'rb'))

	with open(path) as fn:

		lines = []

		for line in fn:
			line = line.rstrip()
			
			if(line):
				lines.append(line)
			else:
				temp_words, temp_roots, temp_features = get_everything(lines) # get every feature of each word of entire sentence
				words.append(temp_words)
				roots.append(temp_roots)
				features.append(temp_features)
				lines.clear()

	print(len(words), len(roots), len(features))
	# pickle.dump(words, open(dump_path+"train_words", 'wb'))
	# pickle.dump(roots, open(dump_path+"train_roots", 'wb'))
	# pickle.dump(features, open(dump_path+"train_features", 'wb'))

	showSentenceWiseFeatures()
	
	showTypesOfFeatures()

