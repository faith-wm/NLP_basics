
#pos tagging with nltk
import nltk
sentence='the patient has a headache'
tokens=nltk.word_tokenize(sentence)
pos_tagged_text=nltk.pos_tag(tokens)
print(pos_tagged_text)



#pos tagging with spacy
import spacy
nlp=spacy.load('en')
sentence=nlp(sentence)

pos_tagged_text=[(i, i.tag_) for i in sentence]
print(pos_tagged_text)





