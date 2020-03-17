

#copied from https://likegeeks.com/nlp-tutorial-using-python-nltk/

import nltk
from nltk.corpus import stopwords
import urllib.request
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt



#get some text from a webpage and analyze it
response=urllib.request.urlopen('http://php.net/')
html=response.read()

#use beautiful soup to clean the html tags
soup=BeautifulSoup(html, features="html.parser")
text=soup.get_text(strip=True)

#tokenixe the test
tokens=[t for t in text.split()]

#calculate tokens frequency distribution using nltk freqDist()
freq=nltk.FreqDist(tokens)
for key, value in freq.items():
    print('{} : {}'.format(str(key), str(value)))

#plot of the tokens
freq.plot(20,cumulative=False)


#remove stopwords
clean_tokens=tokens[:]
stopword_list=stopwords.words('english')

for token in tokens:
    if token in stopword_list:
        clean_tokens.remove(token)

#freq distribution of cleaned tokens
freq=nltk.FreqDist(clean_tokens)
freq.plot(20, cumulative=False)
