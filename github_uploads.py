

#copied from https://likegeeks.com/nlp-tutorial-using-python-nltk/

from nltk.corpus import wordnet


#get synonyms
synonym=wordnet.synsets('pain')
print(synonym)  #returns a list of possible synonyms

for i in range(len(synonym)):     #wordNet gives a brief descriprion of each synonym and example usage
    print(synonym[i].definition())
    print(synonym[i].examples())
    print('----------------------------')


#another way of getting synonyms
synonyms=[]
for synonym in wordnet.synsets('pain'):
    for lemma in synonym.lemmas():
        synonyms.append(lemma.name())
print('\n synonyms: ',synonyms)

#get antonyms
antonyms=[]
for synonym in wordnet.synsets('pain'):
    for lemma in synonym.lemmas():
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())
print('\n antonyms: ', antonyms)