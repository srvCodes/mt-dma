import nltk

def getCorrectMatches(X,y):
	cnt = []

	for i,j in zip(X,y):
			cnt.append(i==j)

	return cnt

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

files = ["feature0context_out.txt","feature1context_out.txt","feature2context_out.txt","feature3context_out.txt","feature4context_out.txt","feature5context_out.txt", "multitask_context_out.txt"]	
data_file = "/home/saurav/Documents/hindi_morph_analysis/outputs/freezing_with_luong_old/"


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
		res.append(getCorrectMatches(i,j))

	res1, res2, res3, res4, res5, res6, res7 = res

	res = []
	for a,b,c,d,e,f,g in zip(res1, res2, res3, res4, res5, res6, res7):
		# print(a and b and c and d and e and f and g)
		res.append(a and b and c and d and e and f and g)

	return len(X[0]), res

# X, y = load_data(data_file)

# res = getCorrectMatches(X,y)

cnt, res = for_all_files()

print("Total test instances: " + str(cnt) + " Total correct instances: " + str(res.count(True)))
