import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from deap import creator, base, tools, algorithms
from scoop import futures
import random
import numpy
from scipy import interpolate
import matplotlib.pyplot as plt
from predict_with_features import returnTrainTestSetsForGA
import pickle 

n_features, X_train, X_test, X_validation, y_train, y_test, y_validation = returnTrainTestSetsForGA()
# pickle.dump(X_train, open('X_trainfe', 'wb'))
# pickle.dump(X_test, open('X_testfe', 'wb'))
# pickle.dump(X_validation, open('X_validationfe', 'wb'))
# pickle.dump(y_train, open('y_trainfeature', 'wb'))
# pickle.dump(y_test, open('y_testfeature', 'wb'))
# pickle.dump(y_validation, open('y_valfeature', 'wb'))

# X_train = pickle.load(open('X_trainfe', 'rb'))
# X_test = pickle.load(open('X_testfe', 'rb'))
# X_validation = pickle.load(open('X_validationfe', 'rb'))
# y_train = pickle.load(open('y_trainfeature', 'rb'))
# y_test = pickle.load(open('y_testfeature', 'rb'))
# y_validation = pickle.load(open('y_valfeature', 'rb'))

X_train = X_train[:56250]
X_test = X_test[:25000]
X_validation = X_validation[:18750]
y_train = y_train[:56250]
y_test = y_test[:25000]
y_validation = y_validation[:18750]

X_trainAndTest = numpy.concatenate((X_train, X_test), axis=0)
print(y_train.shape)
print(y_test.shape)
y_trainAndTest = numpy.concatenate((y_train, y_test), axis=0)

totalPopulation = len(X_train) + len(X_test) + len(X_validation)
allFeatures = numpy.concatenate((X_trainAndTest, X_validation), axis=0)

allFeatures = pd.DataFrame(allFeatures)
allFeatures.columns = ['sentence_len', 'is_first', 'is_last', 'word', 'prefix1', 'prefix2', \
                       'prefix3', 'suffix1', 'suffix2', 'suffix3', 'suffix4',\
                       'previous_word', 'next_word', 'total_vowels', 'nuktas', 'total_punctuations', \
                       'total_numbers', 'total_consonants', 'is_voiced_aspirated', \
						'is_voiceless_aspirated', 'is_modifier', 'is_diphthong', 'is_labiodental', 'is_dental', 'is_glottal',\
						'is_samvrit', 'is_ardhsam', 'is_ardhviv', 'is_vivrit', 'is_lowmid', 'is_upmid', 'is_lowhigh', 'is_high',\
						'is_dvayostha', 'is_dantya', 'is_varstya', 'is_talavya', 'is_murdhanya', 'is_komaltalavya', 'is_nasikya', \
						'is_sparsha', 'is_parshvika', 'is_prakampi', 'is_sangarshi', 'is_ardhsvar',\
						'is_front', 'is_mid', 'is_back', 'is_long', 'is_short', 'is_medium', 'is_dravidian', 'is_bangla', 'is_hard'\
					]


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
    tck = interpolate.splrep(percentileList, testAccuracyList, s=5.0)
    ynew = interpolate.splev(percentileList, tck)
    # plt.show(block=True)
    # plt.interactive(False)

    e = plt.figure(1)
    plt.plot(percentileList, testAccuracyList, marker='o', color='g')
    plt.plot(percentileList, ynew, color='b')
    # plt.title('Validation Set Classification Accuracy vs. \n Continuum with Cubic-Spline Interpolation')
    plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
    plt.ylabel('Validation Set Accuracy')
    e.show()
    


    f = plt.figure(2)
    plt.scatter(percentileList, validationAccuracyList)
    # plt.title('Validation Set Classification Accuracy vs. Continuum')
    plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
    plt.ylabel('Validation Set Accuracy')
    f.show()

    g = plt.figure(3)
    plt.scatter(percentileList, testAccuracyList)
    # plt.title('Test Set Classification Accuracy vs. Continuum')
    plt.xlabel('Population Ordered By Increasing Test Set Accuracy')
    plt.ylabel('Test Set Accuracy')
    g.show()
    e.savefig('1.png')
    f.savefig('2.png')
    g.savefig('3.png')

    # input()