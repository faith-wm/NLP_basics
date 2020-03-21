
#parts of this code are copied from
# https://towardsdatascience.com/a-short-introduction-to-nlp-in-python-with-spacy-d0aa819af3ad and
# https://realpython.com/natural-language-processing-spacy-python/#visualization-using-displacy

import spacy
from spacy import displacy
nlp = spacy.load("en")  #if getting error try:  pip3 install spacy && python3 -m spacy download en

wiki_text = """GitHub, Inc. is a US-based global company that provides hosting for software development 
version control using Git. It is a subsidiary of Microsoft, which acquired the company in 2018 for
 US$7.5 billion. It offers the distributed version control and source code management (SCM) functionality 
 of Git, plus its own features. It provides access control and several collaboration features such as bug 
 tracking, feature requests, task management, and wikis for every project."""

wiki_text = nlp(wiki_text)

#spacy also has noun phrase detection for extracting noun phrases
print('\n===================')
for chunk in wiki_text.noun_chunks:
    print(chunk)


#NER parsing
print('\n===================')
parsed_text=[(i, i.label_, i.label) for i in wiki_text.ents]
print(parsed_text)          #returns named entities

#dependency parsing to learn the relationships between words
print('\n======================')
for token in wiki_text:
    print ('{}:  {}:  {}:  {}  '.format(token.text, token.tag_, token.head.text, token.dep_))


#visualization the dependency parsing
displacy.serve(wiki_text, style='dep')  #gives a web server where one can see the dependency tree in browser
                                        #style=dep shows POS tags in a graph-like visualization
displacy.serve(wiki_text, style='ent')  #style=ent highlights named entities and labels in a text


