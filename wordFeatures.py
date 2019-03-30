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
##    
##documents= [(list(movie_reviews.words(fileid)),category)
##            for category in movie_reviews.categories()
##            for fileid in movie_reviews.fileids(category)]
##
##random.shuffle(documents)
##allwords = []
##
##for w in movie_reviews.words():
##    allwords.append(w.lower())
##


short_pos = open("short_reviews/positivefull.txt","r").read()
short_neg = open("short_reviews/negativefull.txt","r").read()


documents =[]
all_words =[]

##j is adjective , r is an adverd and v is a verb
##allowed_word_types =["J","R","V"]

allowed_word_types = ["V"]

for r in short_pos.split('\n'):
    documents.append((r,"pos"))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):
    documents.append((r,"neg"))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


save_classifier = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents,save_classifier)
save_classifier.close()
    
all_words = nltk.FreqDist(all_words)

print(all_words.most_common(10))

word_features  = list(all_words.keys())[:5000]

save_wordfeatures = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features,save_wordfeatures)
save_wordfeatures.close()

def findwords(document):
    words = word_tokenize(document)
    features={}
    for i in word_features:
        features[i] = (i in words)

    return features

#print ((findwords(movie_reviews('pos\cv000_295950.txt'))))
#CategorizedPlainTextCorpusReader is not callable error occuring
#else everything working perfectly


featuresets = [(findwords(rev),category) for (rev,category) in documents]
random.shuffle(featuresets)
print(len(featuresets))
#simple print of the featuresets take s alot of time to print

training_sets = featuresets[:10000]
testing_sets = featuresets[10000:]



##classifier_f = open("naivebayes.pickle","rb")
##classifier = pickle.load(classifier_f)
##classifier_f.close()

classifier  = nltk.NaiveBayesClassifier.train(training_sets)
print (" Original Naive Bayes algo accuracy percent :: ",(nltk.classify.accuracy(classifier,testing_sets)*100))
classifier.show_most_informative_features(15)


classifier_save = open("pickled_algos/originalnaivebayes.pickle","wb")
pickle.dump(classifier,classifier_save)
classifier_save.close()

#save_classifier = open("naivebayes.pickle","wb")
#pickle.dump(classifier,save_classifier)
#save_classifier.close()

MultinomialNB_Classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_Classifier.train(training_sets)
print ("MNB_CLassifier Accuracy percent:", nltk.classify.accuracy(MultinomialNB_Classifier,testing_sets)*100) 

MNBclassifier_save = open("pickled_algos/multinomialnaivebayes.pickle","wb")
pickle.dump(MultinomialNB_Classifier,MNBclassifier_save)
MNBclassifier_save.close()

##GaussianNB_Classifier = SklearnClassifier(GaussianNB())
##GaussianNB_Classifier.train(training_sets)
##print ("GaussianNB_CLassifier Accuracy percent:", nltk.classify.accuracy(GaussianNB_Classifier,testing_sets)*100) 

BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_Classifier.train(training_sets)
print ("BernoulliNB_CLassifier Accuracy percent:", nltk.classify.accuracy(BernoulliNB_Classifier,testing_sets)*100)

classifier_save = open("pickled_algos/BernoulliNBnaivebayes.pickle","wb")
pickle.dump(BernoulliNB_Classifier,classifier_save)
classifier_save.close()


#LogisticRegression, SGDClassifier
#SVC , LinearSVC , NuSVC

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_sets)
print ("LogisticRegression_CLassifier Accuracy percent:", nltk.classify.accuracy(LogisticRegression_Classifier,testing_sets)*100)

classifier_save = open("pickled_algos/logisticregressionnaivebayes.pickle","wb")
pickle.dump(LogisticRegression_Classifier,classifier_save)
classifier_save.close()


SGDclassifier = SklearnClassifier(SGDClassifier())
SGDclassifier.train(training_sets)
print ("SGDClassifier_CLassifier Accuracy percent:", nltk.classify.accuracy(SGDclassifier,testing_sets)*100)

classifier_save = open("pickled_algos/SGDnaivebayes.pickle","wb")
pickle.dump(SGDclassifier,classifier_save)
classifier_save.close()

##
##SVCclassifier = SklearnClassifier(SVC())
##SVCclassifier.train(training_sets)
##print ("SVCclassifier Accuracy percent:", nltk.classify.accuracy(SVCclassifier,testing_sets)*100)

linearSVCclassifier = SklearnClassifier(LinearSVC())
linearSVCclassifier.train(training_sets)
print ("linearSVCclassifier_CLassifier Accuracy percent:", nltk.classify.accuracy(linearSVCclassifier,testing_sets)*100)

classifier_save = open("pickled_algos/LinearSVCnaivebayes.pickle","wb")
pickle.dump(linearSVCclassifier,classifier_save)
classifier_save.close()


NuSVCclassifier = SklearnClassifier(NuSVC())
NuSVCclassifier.train(training_sets)
print ("NuSVCclassifier Accuracy percent:", nltk.classify.accuracy(NuSVCclassifier,testing_sets)*100)

classifier_save = open("pickled_algos/NuSVCnaivebayes.pickle","wb")
pickle.dump(NuSVCclassifier,classifier_save)
classifier_save.close()



voted_classifier = vote(classifier,
                        NuSVCclassifier,
                        linearSVCclassifier,
                        SGDclassifier,
                        MultinomialNB_Classifier,
                        BernoulliNB_Classifier,
                        LogisticRegression_Classifier)

print ("vote classifier accuracy percent :",(nltk.classify.accuracy(voted_classifier,testing_sets))*100)
##
##print ("Classification :", voted_classifier.classify(testing_sets[0][0]), "Confidence :: ", voted_classifier.confidence(testing_sets[0][0])*100)
##print ("Classification :", voted_classifier.classify(testing_sets[1][0]), "Confidence :: ", voted_classifier.confidence(testing_sets[0][0])*100)
##print ("Classification :", voted_classifier.classify(testing_sets[2][0]), "Confidence :: ", voted_classifier.confidence(testing_sets[0][0])*100)
##print ("Classification :", voted_classifier.classify(testing_sets[3][0]), "Confidence :: ", voted_classifier.confidence(testing_sets[0][0])*100)
##print ("Classification :", voted_classifier.classify(testing_sets[4][0]), "Confidence :: ", voted_classifier.confidence(testing_sets[0][0])*100)


def sentiment(text):
    feats = findwords(text)

    return voted_classifier.classify(feats)

