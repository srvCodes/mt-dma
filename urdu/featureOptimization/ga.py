import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from load_data_with_phonetic import load_data_for_features

from deap import creator, base, tools, algorithms
from scoop import futures
import random
import numpy
from scipy import interpolate
import matplotlib.pyplot as plt
from predict_with_features import returnTrainTestSets
import pickle 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter, deque
from keras.utils import np_utils

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
            enc[i].fit(y[i])
            transformed.append(enc[i].transform(y[i]))

    else:
        transformed = []
        print("processing test encoders !")
        for i in range(len(y)):
            transformed.append(enc[i].transform(y[i]))

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

    print("Encoders dumped ################################################")
    return (y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6, enc, labels)

path = './pickle_dumps/'
n_features, X_train, X_test, X_validation = returnTrainTestSets()
features = pickle.load(open('./pickle_dumps/test_features', 'rb'))
y1, y2, y3, y4, y5, y6 = load_data_for_features(features)
y1, n1, y2, n2, y3, n3, y4, n4, y5, n5, y6, n6, enc, labels = process_features(y1, y2, y3, y4, y5, y6)


pickle.dump(X_test, open(path+'X_ga', 'wb'))
pickle.dump(y1, open(path+'y1_ga', 'wb'))
pickle.dump(y2, open(path+'y2_ga', 'wb'))
pickle.dump(y3, open(path+'y3_ga', 'wb'))
pickle.dump(y4, open(path+'y4_ga', 'wb'))
pickle.dump(y5, open(path+'y5_ga', 'wb'))
pickle.dump(y6, open(path+'y6_ga', 'wb'))


X = pickle.load(open(path+'X_ga', 'rb'))
y = pickle.load(open(path+'y1_ga', 'rb'))
train_test_cutoff = int(.75 * len(X))   
y = y[:len(X)] 

X_train = X[:train_test_cutoff]
X_test = X[train_test_cutoff:]
y_train = y[:train_test_cutoff]
y_test = y[train_test_cutoff:]

train_val_cutoff = int(.75 * len(X_train))
X_train = X_train[:train_val_cutoff]
X_validation = X_train[train_val_cutoff:]
y_train = y_train[:train_val_cutoff]
y_validation = y_train[train_val_cutoff:]

X_trainAndTest = numpy.concatenate((X_train, X_test), axis=0)
print(y_train.shape)
print(y_test.shape)
y_trainAndTest = numpy.concatenate((y_train, y_test), axis=0)

totalPopulation = len(X_train) + len(X_test) + len(X_validation)
allFeatures = numpy.concatenate((X_trainAndTest, X_validation), axis=0)

allFeatures = pd.DataFrame(allFeatures)
allFeatures.columns = ['length', 'is_first', 'is_last', 'prefix1', 'prefix2', 'prefix3', 'suffix1', 'suffix2', 'suffix3', 'suffix4', 'prev_word', 'next_word',\
            'total_vowels', 'total_numbers', 'total_modifiers', 'total_consonants', 'is_aspirated',  \
            'is_longer', 'is_shorter', 'is_front', 'is_mid', 'is_back', 'is_diph', 'is_dravidian', 'is_unvoiced', 'is_hard',\
            'is_n', 'is_v', 'is_nasikya', 'is_sparsha', 'is_parshvika', 'is_prakampi', 'is_sangarshi', 'is_ardhsvar',\
            'is_dvayostha', 'is_dantya', 'is_varstya', 'is_talavya', 'is_murdhanya', 'is_komaltalavya',\
            'is_l', 'is_lm', 'is_um', 'is_g', 'is_h', 'is_s', 'is_m', 'is_i', 'is_v2']


def scorer(clf, X_test, y_test):
    predictions = clf.predict(X_test)
    f1 = f1_score(y_test, predictions, average='micro')

    return f1


# Feature subset fitness function
def getFitness(individual, X_train, X_test, y_train, y_test):
    # Apply logistic regression on the data, and calculate accuracy
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    f1_scores = scorer(clf, X_test, y_test)
    # Return calculated accuracy as fitness
    return (f1_scores.mean(),)

# ========DEAP GLOBAL VARIABLES (viewable by SCOOP)========

# Create Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(allFeatures.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Continue filling toolbox...
toolbox.register("evaluate", getFitness, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


# ========

def getHof():
    # Initialize variables to use eaSimple
    numPop = 60
    numGen = 30
    pop = toolbox.population(n=numPop)
    hof = tools.HallOfFame(numPop * numGen)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Launch genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.03, ngen=numGen, stats=stats, halloffame=hof,
                                   verbose=True)

    # Return the hall of fame
    return hof


def getMetrics(hof):
    # Get list of percentiles in the hall of fame
    percentileList = [i / (len(hof) - 1) for i in range(len(hof))]

    # Gather fitness data from each percentile
    testAccuracyList = []
    validationAccuracyList = []
    individualList = []
    for individual in hof:
        testAccuracy = individual.fitness.values
        validationAccuracy = getFitness(individual, X_trainAndTest, X_validation, y_trainAndTest, y_validation)
        testAccuracyList.append(testAccuracy[0])
        validationAccuracyList.append(validationAccuracy[0])
        individualList.append(individual)
    testAccuracyList.reverse()
    validationAccuracyList.reverse()
    return testAccuracyList, validationAccuracyList, individualList, percentileList


if __name__ == '__main__':

    '''
    First, we will apply logistic regression using all the features to acquire a baseline accuracy.
    '''
    print("length is: ", len(X_train[1]))
    individual = [1 for i in range(len(X_train[1]))]
    testAccuracy = getFitness(individual, X_train, X_test, y_train, y_test)
    validationAccuracy = getFitness(individual, X_trainAndTest, X_validation, y_trainAndTest, y_validation)
    print('\nTest accuracy with all features: \t' + str(testAccuracy[0]))
    print('Validation accuracy with all features: \t' + str(validationAccuracy[0]) + '\n')

    '''
    Now, we will apply a genetic algorithm to choose a subset of features that gives a better accuracy than the baseline.
    '''
    hof = getHof()
    testAccuracyList, validationAccuracyList, individualList, percentileList = getMetrics(hof)

    # Get a list of subsets that performed best on validation data
    maxValAccSubsetIndicies = [index for index in range(len(validationAccuracyList)) if
                               validationAccuracyList[index] == max(validationAccuracyList)]
    maxValIndividuals = [individualList[index] for index in maxValAccSubsetIndicies]
    maxValSubsets = [[list(allFeatures)[index] for index in range(len(individual)) if individual[index] == 1] for
                     individual in maxValIndividuals]

    print('\n---Optimal Feature Subset(s)---\n')
    for index in range(len(maxValAccSubsetIndicies)):
        print('Percentile: \t\t\t' + str(percentileList[maxValAccSubsetIndicies[index]]))
        print('Validation Accuracy: \t\t' + str(validationAccuracyList[maxValAccSubsetIndicies[index]]))
        print('Individual: \t' + str(maxValIndividuals[index]))
        print('Number Features In Subset: \t' + str(len(maxValSubsets[index])))
        print('Feature Subset: ' + str(maxValSubsets[index]))

    '''
    Now, we plot the test and validation classification accuracy to see how these numbers change as we move from our worst feature subsets to the 
    best feature subsets found by the genetic algorithm.
    '''
    # Calculate best fit line for validation classification accuracy (non-linear)
    tck = interpolate.splrep(percentileList, validationAccuracyList, s=5.0)
    ynew = interpolate.splev(percentileList, tck)
    # plt.show(block=True)
    # plt.interactive(False)

    e = plt.figure(1)
    plt.plot(percentileList, validationAccuracyList, marker='o', color='r')
    plt.plot(percentileList, ynew, color='b')
    plt.title('Validation Set Classification Accuracy vs. \n Continuum with Cubic-Spline Interpolation')
    plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
    plt.ylabel('Validation Set Accuracy')
    e.show()


    f = plt.figure(2)
    plt.scatter(percentileList, validationAccuracyList)
    plt.title('Validation Set Classification Accuracy vs. Continuum')
    plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
    plt.ylabel('Validation Set Accuracy')
    f.show()

    g = plt.figure(3)
    plt.scatter(percentileList, testAccuracyList)
    plt.title('Test Set Classification Accuracy vs. Continuum')
    plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
    plt.ylabel('Test Set Accuracy')
    g.show()

    input()