import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB , GaussianNB , BernoulliNB 
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC , LinearSVC , NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class vote(ClassifierI):
    def __init__(self , *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)

        return conf

save_classifier = open("pickled_algos/documents.pickle","rb")
documents = pickle.load(save_classifier)
save_classifier.close()

save_wordfeatures = open("pickled_algos/word_features5k.pickle","rb")
word_features = pickle.load(save_wordfeatures)
save_wordfeatures.close()

def findwords(document):
    words = word_tokenize(document)
    features={}
    for i in word_features:
        features[i] = (i in words)

    return features


featuresets = [(findwords(rev),category) for (rev,category) in documents]
random.shuffle(featuresets)
print(len(featuresets))

training_sets = featuresets[:10000]
testing_sets = featuresets[10000:]

classifier_save = open("pickled_algos/originalnaivebayes.pickle","rb")
classifier = pickle.load(classifier_save)
classifier_save.close()


MNBclassifier_save = open("pickled_algos/multinomialnaivebayes.pickle","rb")
MultinomialNB_Classifier = pickle.load(MNBclassifier_save)
MNBclassifier_save.close()


classifier_save = open("pickled_algos/BernoulliNBnaivebayes.pickle","rb")
BernoulliNB_Classifier = pickle.load(classifier_save)
classifier_save.close()


classifier_save = open("pickled_algos/logisticregressionnaivebayes.pickle","rb")
LogisticRegression_Classifier = pickle.load(classifier_save)
classifier_save.close()


classifier_save = open("pickled_algos/SGDnaivebayes.pickle","rb")
SGDclassifier = pickle.load(classifier_save)
classifier_save.close()

classifier_save = open("pickled_algos/LinearSVCnaivebayes.pickle","rb")
linearSVCclassifier = pickle.load(classifier_save)
classifier_save.close()


classifier_save = open("pickled_algos/NuSVCnaivebayes.pickle","rb")
NuSVCclassifier = pickle.load(classifier_save)
classifier_save.close()

##linearSVC classifier not working as there is no data being stored in the respective
##pickle file created for it.Therefore, it is not used else if it works it would more precise the output

voted_classifier = vote(classifier,
                        NuSVCclassifier,
                        SGDclassifier,
                        linearSVCclassifier,
                        MultinomialNB_Classifier,
                        BernoulliNB_Classifier,
                        LogisticRegression_Classifier)

def sentiment(text):
    feats = findwords(text)

    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

