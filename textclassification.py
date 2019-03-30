import nltk
import random
from nltk.corpus import movie_reviews

documents= [(list(movie_reviews.words(fileid)),category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

#print (documents[100])
allwords = []

for w in movie_reviews.words():
    allwords.append(w.lower())


allword = nltk.FreqDist(allwords)
#print(allword[w]
 #     for i in movie_reviews.words() if len(i) > 15
  #    for w in allwords[i])

vocab = allword.keys()
len15 = {}

for i in vocab:
    if len(i) == 20 :
        len15[i] = allword[i]

print (sorted(len15.items(), key=lambda x:x[1],reverse = True))

