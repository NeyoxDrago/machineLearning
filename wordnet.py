from nltk.corpus import wordnet

s = wordnet.synsets('good')

print(s[0])
print(s[0].lemmas()[0].name())

print(s[0].definition())

print (s[0].examples())

synonyms =[]
antonyms = []

for i in s:
    for l in i.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))



w1 = wordnet.synset('good.n.01')
w2 = wordnet.synset('ill.n.01')

print (w1.wup_similarity(w2))
print (w1.path_similarity(w2))

w1 = wordnet.synsets('dog')
w2 = wordnet.synsets('cat')

print (w1[0].wup_similarity(w2[0])) 
