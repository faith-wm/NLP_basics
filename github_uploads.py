
#string similarity measures
import numpy as np
import nltk
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
import os
import py_stringmatching as sm
from scipy import spatial

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.nist_score import sentence_nist
import rouge

import gensim

s1='the patient is sick again'
s2='the patient will was advice to come for tomorrow again'


def sentence_lemmatization(sentence_tokens):
    lemmatizer=WordNetLemmatizer()
    lemmatized_sentence=[]
    for token in sentence_tokens:
        lemmatized_word=lemmatizer.lemmatize(token)
        lemmatized_sentence.append(lemmatized_word)

    return lemmatized_sentence


def ngram_overlap(s1,s2,grams):
    s1_tokens=nltk.word_tokenize(s1)
    s2_tokens=nltk.word_tokenize(s2)

    #lemmatize the tokens==optiona;
    s1_tokens=sentence_lemmatization(s1_tokens)
    s2_tokens=sentence_lemmatization(s2_tokens)

    s1_ngrams=list(ngrams(s1_tokens,grams))
    s2_ngrams=list(ngrams(s2_tokens,grams))

    common_ngrams=set(s1_ngrams+s2_ngrams)
    overlap=2*((len(s1_tokens)/len(common_ngrams))+(len(s2_tokens)/len(common_ngrams)))
    return overlap


def longestCommonPrefix(s1,s2):
    s1_tokens=nltk.word_tokenize(s1)
    s2_tokens=nltk.word_tokenize(s2)

    #lemmatize
    s1=' '.join(sentence_lemmatization(s1_tokens))
    s2=' '.join(sentence_lemmatization(s2_tokens))

    longestCommonPrefix=os.path.commonprefix([s1,s2])
    return longestCommonPrefix

def longestCommonSuffix(s1,s2):
    s1_tokens = nltk.word_tokenize(s1)
    s2_tokens = nltk.word_tokenize(s2)

    # lemmatize
    s1 = ' '.join(sentence_lemmatization(s1_tokens))
    s2 = ' '.join(sentence_lemmatization(s2_tokens))

    #here i reverse the string order and use the method for longest common prefix
    reverse_strings=[' '.join(s.split()[::-1]) for s in [s1,s2]]
    reversed_longestCommonSuffix = os.path.commonprefix(reverse_strings)

    longestCommonSuffix=' '.join(reversed_longestCommonSuffix.split()[::-1])
    return longestCommonSuffix

#longest common substring, longest common sequence

def string_similarity(s1,s2):
    s1_tokens=nltk.word_tokenize(s1)
    s2_tokens=nltk.word_tokenize(s2)

    #take input as tokens
    jaccard = sm.Jaccard()
    jaccard=jaccard.get_raw_score(s1_tokens,s2_tokens)

    cosine = sm.Cosine()
    cosine=cosine.get_raw_score(s1_tokens,s2_tokens)

    dice = sm.Dice()
    dice=dice.get_raw_score(s1_tokens,s2_tokens)

    overlap = sm.OverlapCoefficient()
    overlap=overlap.get_raw_score(s1_tokens,s2_tokens)

    tfidf = sm.TfIdf()
    tfidf=tfidf.get_raw_score(s1_tokens,s2_tokens)

    tsversky = sm.TverskyIndex()
    tsversky=tsversky.get_raw_score(s1_tokens,s2_tokens)

    generalizeJaccard = sm.GeneralizedJaccard()
    generalizeJaccard=generalizeJaccard.get_raw_score(s1_tokens,s2_tokens)

    mongeElkan = sm.MongeElkan()
    mongeElkan=mongeElkan.get_raw_score(s1_tokens,s2_tokens)


    #take string as input
    affine = sm.Affine()
    affine=affine.get_raw_score(s1,s2)

    bagdistance = sm.BagDistance()
    bagdistance=bagdistance.get_raw_score(s1,s2)

    editex = sm.Editex()
    editex=editex.get_raw_score(s1,s2)

    jaro = sm.Jaro()
    jaro=jaro.get_raw_score(s1,s2)

    jaroWinkler = sm.JaroWinkler()
    jaroWinkler=jaroWinkler.get_raw_score(s1,s2)

    levenhtein = sm.Levenshtein()
    levenhtein=levenhtein.get_raw_score(s1,s2)

    needlemanWunsch = sm.NeedlemanWunsch()
    needlemanWunsch=needlemanWunsch.get_raw_score(s1,s2)

    smithWaterman = sm.SmithWaterman()
    smithWaterman=smithWaterman.get_raw_score(s1,s2)



    return [jaccard, cosine]


def Machine_Translation_score(s1,s2):
    s1_tokens=nltk.word_tokenize(s1)
    s2_tokens=nltk.word_tokenize(s2)

    bleu=sentence_bleu([s1_tokens],s2_tokens)
    gleu=sentence_gleu([s1_tokens], s2_tokens)
    nist=sentence_nist([s1_tokens], s2_tokens)

    #rouge
    evaluator = rouge.Rouge(metrics=['rouge-l'], max_n=4, limit_length=True, length_limit=100,
                            length_limit_type='words', alpha=0.5,
                            weight_factor=1.2, stemming=True)
    scores = evaluator.get_scores(s1, s2)
    for x in scores:
        for y in scores[x]:
            if y == "f":
                rouge_score = scores[x][y]

    return [bleu, gleu, nist, rouge_score]



def average_feature_vector(sentence, model, num_features, index2word_set):  #get average of word embeddings
    tokens=nltk.word_tokenize(sentence)
    feature_vector=np.zeros((num_features), dtype='float32')
    num_words=0

    for word in tokens:
        if word in index2word_set:
            num_words+=1
            feature_vector=np.add(feature_vector, model[word])
    if (num_words>0):
        feature_vector=np.divide(feature_vector, num_words)

    return feature_vector




def word2VecFeature(embedding_dim,embedding_location,sentence, model, index2word_set):

    sentence_avg_feature=average_feature_vector(sentence,model=model,num_features=embedding_dim,index2word_set=index2word_set)

    return sentence_avg_feature


def embedding_based_similarity(s1,s2, embedding_dim,embedding_location):
    print('please wait loading model takes a few minutes')
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_location, binary=True)
    index2word_set = set(model.wv.index2word)

    s1_featureVec= word2VecFeature(embedding_dim=embedding_dim, embedding_location=embedding_location, sentence=s1,
                                   model=model, index2word_set=index2word_set)
    s2_featureVec = word2VecFeature(embedding_dim=300, embedding_location=embeddings_path, sentence=s2,
                                    model=model, index2word_set=index2word_set)

    cosine = 1 - spatial.distance.cosine(s1_featureVec,s2_featureVec)
    euclidean = 1 - spatial.distance.euclidean(s1_featureVec, s2_featureVec)
    squareEuclidean = 1 - spatial.distance.sqeuclidean(s1_featureVec, s2_featureVec)
    correlation = 1 - spatial.distance.correlation(s1_featureVec, s2_featureVec)

    return [cosine,euclidean,squareEuclidean,correlation]



# embeddings_path="/Users/..../GoogleNews-vectors-negative300.bin.gz"
# print(embedding_based_similarity(s1,s2,300,embeddings_path))
