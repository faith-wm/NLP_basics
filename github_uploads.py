

#copied from https://likegeeks.com/nlp-tutorial-using-python-nltk/

from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer


supported_languages=SnowballStemmer.languages
print(supported_languages)


#stemming english words
stemmer=PorterStemmer()
print(stemmer.stem('speaking'))


#lemmatize words using WordNet
lemmatizer=WordNetLemmatizer()
print(lemmatizer.lemmatize('speaking'))

#lemmatizing the word speaking results in speaking
#in lemmatizer we can specify that the result should be a verb, noun, adjective, adverb
print(lemmatizer.lemmatize('speaking',pos='v'))
print(lemmatizer.lemmatize('speaking',pos='n'))
print(lemmatizer.lemmatize('speaking',pos='a'))
print(lemmatizer.lemmatize('speaking',pos='r'))