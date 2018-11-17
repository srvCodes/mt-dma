from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import nltk
import os
import sys 

#print(fuzz.ratio("this is a test", "this is a pre-test!"))
def lev_dist(source, target):
    if source == target:
        return 0


    # Prepare a matrix
    slen, tlen = len(source), len(target)
    dist = [[0 for i in range(tlen+1)] for x in range(slen+1)]
    for i in range(slen+1):
        dist[i][0] = i
    for j in range(tlen+1):
        dist[0][j] = j

    # Counting distance, here is my function
    for i in range(slen):
        for j in range(tlen):
            cost = 0 if source[i] == target[j] else 1
            dist[i+1][j+1] = min(
                            dist[i][j+1] + 1,   # deletion
                            dist[i+1][j] + 1,   # insertion
                            dist[i][j] + cost   # substitution
                        )
    return dist[-1][-1]

def load_data(data_file):
	text = open(data_file, 'r', encoding="utf-8").readlines()[1:]

	word_list = []

	for line in text:
		line = line.strip()
		stripped_line = line.replace('\u200b','')
		stripped_line = line.replace('\u200d','').split('\t\t')
		word_list.append(stripped_line)
		#print(word_list)
		
	X = [c[1] for c in word_list]
	y = [c[2] for c in word_list]

	y = [i.lstrip() for i in y]
	print(y)
	# for i,j in zip(X,y):
	# 	print(i + '\t' + j)

	X = [list(x) for x, w in zip(X, y) if len(x) > 0 and len(w) > 0] # list of lists
	y = [list(w) for x, w in zip(X,y) if len(x) > 0 and len(w) > 0]

	return (X, y)
	
def word_acc(X, y):
	res = [int(i==j) for (i,j) in zip(X,y)]

	return res.count(1)

data_file = "/home/saurav/Documents/hindi_morph_analysis/outputs/freezing_with_luong_old/multitask_context_out.txt"

X, y = load_data(data_file)

sum = 0
for i,j in zip(X,y):
	sum += lev_dist(i,j)

res = sum/len(X)
print("Leveishtein similarity of whole doc: ", res)


res = word_acc(X,y)
print("Word accuracy: ", res/len(X) * 100) # 93.668