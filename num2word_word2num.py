
from num2words import num2words
from words2num import words2num
import re
import inflect




def number_to_word(number):
    model=inflect.engine()
    return model.number_to_words(number)

# def number_to_word(number):
#     return num2words(number)

def word_to_number(word):
    return words2num(word)


def word_to_number_replacement(sentence):
    tokens=sentence.split()
    for word in tokens:
        try:
            number=word_to_number(word)
            tokens=[str(number) if token==word else token for token in tokens]
        except:
            pass
    return ' '.join(tokens)


def number_to_word_replacement(sentence):
    tokens=sentence.split()
    for word in tokens:
        if word.isdigit() or re.match("^\d+?\.\d+?$", word) is not None:  #re.match if for numbers with decimals
            num2word=number_to_word(word)
            tokens = [num2word if token == word else token for token in tokens]
    return ' '.join(tokens)




print(number_to_word('20'))
print(number_to_word('20.40'))
print(word_to_number('thirty five'))
print(word_to_number_replacement('i am twenty-four years and my brother is thirty '))
print(number_to_word_replacement('i am 24.5 and my brother is 30 years'))


