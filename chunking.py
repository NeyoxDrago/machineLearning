import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

text = state_union.raw('1963-Kennedy.txt')
custom_tokenizer = PunktSentenceTokenizer()

tokenizedtext = custom_tokenizer.tokenize(text)

def getContent():
    try:
        for i in tokenizedtext[:1]:
            words = nltk.word_tokenize(i)
            tag = nltk.pos_tag(words)

            chunkgram = r"""hub :{<NNP.?>*<RB.?>*}"""

            chinkgram = r"""hub :{<.*>}
                                }<NNP.?>{"""

            parser = nltk.RegexpParser(chinkgram)
            chunked = parser.parse(tag)

            #print(chunked)
            chunked.draw()

    except Exception as e:
        print(str(e))

def namedentity():
    try:
        for i in tokenizedtext[:1]:
            words = nltk.word_tokenize(i)
            tag = nltk.pos_tag(words)

            nameentity = nltk.ne_chunk(tag,binary = True)

            nameentity.draw()
    except Exception as e:
        print(str(e))


namedentity()
