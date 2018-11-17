import nltk

def getCorrectMatches(X,y):
	cnt = []

	for i,j in zip(X,y):
			cnt.append(i==j)

	return len(X), cnt

def load_data(data_file):
	text = open(data_file, 'r', encoding="utf-8").readlines()[1:]

	word_list = []

	for line in text:
		line = line.strip()
		stripped_line = line.replace('\u200d','').split('\t\t')
		word_list.append(stripped_line)
		#print(word_list)
		
	X = [c[1] for c in word_list]
	y = [c[2] for c in word_list]

	y = [i.lstrip() for i in y]
	# for i,j in zip(X,y):
	# 	print(i + '\t' + j)

	X = [x for x, w in zip(X, y) if len(x) > 0 and len(w) > 0] # list of lists
	y = [w for x, w in zip(X,y) if len(x) > 0 and len(w) > 0]

	return (X, y)

files = ["feature0context_out1.txt","feature1context_out1.txt","feature2context_out1.txt","feature3context_out1.txt","feature4context_out1.txt","feature5context_out1.txt"]	
data_file = "/home/saurav/Documents/urdu_morph_analysis/outputs/freezing_with_luong/"


def for_all_files():
	X = []
	y = []
	j = 0

	for i in files:
		path = data_file+i
		tmp1, tmp2 = load_data(path)
		X.append(tmp1)
		y.append(tmp2)
		j += 1

	res = []
	for i,j in zip(X,y):
		a,b = getCorrectMatches(i,j)
		res.append(b)

	res1, res2, res3, res4, res5, res6 = res
	print(res1[:10])
	res = []
	for a,b,c,d,e,f in zip(res1, res2, res3, res4, res5, res6):
		# print(a and b and c and d and e and f and g)
		res.append(a and b and c and d and e and f)

	return len(X[0]), res


# X, y = load_data(data_file+files[6])
# print(y[:10])
# cnt, res = getCorrectMatches(X,y)

cnt, res = for_all_files()
print("Total test instances: " + str(cnt) + " Total correct instances: " + str(res.count(True)))
