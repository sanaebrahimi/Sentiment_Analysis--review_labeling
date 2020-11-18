#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import statsmodels.api as sm
import itertools
from itertools import combinations, chain
import nltk
from nltk.sem.logic import NegatedExpression
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import math



MINCOUNT = 10

wordpat  = re.compile(r"[A-Za-z]+[A-Za-z']?[A-Za-z]+", re.IGNORECASE)
#word_pat = re.compile(r"[a-z]+[']+[a-z]+", re.IGNORECASE)


stopwords = ["able", "about", "across", "after", "all", "almost", "also", "am", "among",
"an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by",
"can", "cannot", "could", "dear", "did", "do", "does", "either", "else",
"ever", "every", "for", "from", "get", "got", "had", "has", "have", "he",
"her", "hers", "him", "his", "how", "however", "if", "in", "into", "is", "it",
"its", "just", "least", "let", "like", "likely", "may", "me", "might", "most",
"must", "my", "neither", "no", "nor", "of", "off", "often", "on",
"only", "or", "other", "our", "own", "rather", "said", "say", "says", "she",
"should", "since", "so", "some", "than", "that", "the", "their", "them",
"then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us",
"wants", "was", "we", "were", "what", "when", "where", "which", "while", "who",
"whom", "why", "will", "with", "would", "yet", "you", "your"]


CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

stopdict = {}
wordnet_lemmatizer = WordNetLemmatizer()

sentiment_words = {}
twogram_words = defaultdict(lambda : 0)

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    #using a given text and a contraction map, this function handls contractions by expanding them to the complete form
    
    global CONTRACTION_MAP
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def process_dict(s, dict, i):
    global wordpat
    global wordnet_lemmatizer
    global stopwords
    nwords = 0
    tmpdict = defaultdict(lambda: 0)   # counts words in THIS review

    tmp1dict = defaultdict(lambda: 0)   # counts words in THIS review
    Neg_flag = False

    s = expand_contractions(s)
    for w in wordpat.finditer(s):
        x,y = w.span()
        key = wordnet_lemmatizer.lemmatize(s[x:y].lower(), pos="v")

        if y - x > 1 and key not in stopwords:

            if key == 'not':
                Neg_flag = True
                continue
            if Neg_flag:
                tmp1dict[key] += 1
            else:
                tmpdict[key] += 1
            nwords += 1
    for key in tmpdict:

        # add one to words that appear in THIS review
        dict[i][key] += 1
    for key in tmp1dict:
        dict[1-i][key] += 1
    return nwords

def sentiment_word(pathlist):
    global sentiment_words
    d1 = defaultdict(lambda: 0)
    d2 = defaultdict(lambda: 0)
    dictlist = [d1, d2]
    wordcount = [0, 0]

    for i in range(2):
        path = pathlist[i]
        doctree = ET.parse(path)
        root = doctree.getroot()

        for child in root:
            for item in child:
                if item.tag == "review_text":
                    wordcount[i] += process_dict(item.text, dictlist, i)


    wk = list(d1.keys())  # wk == word keys
    wk.extend(d2.keys())
    wk = set(wk)
    wk = list(wk)


    n1 = wordcount[0]
    n2 = wordcount[1]
    sentiment_words = {}
    for k in range(len(wk)):
        k1 = d1[wk[k]]
        k2 = d2[wk[k]]
        if k1 * k2 < 59:
            continue
        z, pv = sm.stats.proportions_ztest([k1, k2], [n1, n2])
        if pv < 0.01:
            sentiment_words[wk[k]] = (d1[wk[k]], d2[wk[k]], pv)
            
    print(len(sentiment_words))



def process_2gram(pathlist):
    global wordpat
    global sentiment_words
    global twogram_words
    global wordnet_lemmatizer

    #print(list_of_subsets[0])
    #for key in sets:
    d1 = defaultdict(lambda: 0)
    d2 = defaultdict(lambda: 0)
    dictlist = [d1, d2]
    wordcount = [0, 0]
    for i in range(2):
        path = pathlist[i]
        doctree = ET.parse(path)
        root = doctree.getroot()

        for child in root:
            for item in child:
                if item.tag == "review_text":
                    l = []
                    for w in wordpat.finditer(item.text):
                        wordcount[i] += 1
                        x, y = w.span()
                        #print(w)
                        k = wordnet_lemmatizer.lemmatize(item.text[x:y].lower(), pos="v")
                        if k in sentiment_words:
                            l.append(k)
                    list_of_subsets = list(map(list, itertools.combinations(l, 2)))
                    for subset in list_of_subsets:
                        dictlist[i][(subset[0], subset[1])] += 1

    wk = list(d1.keys())  # wk == word keys
    wk.extend(d2.keys())
    wk = set(wk)
    wk = list(wk)


    n1 = wordcount[0]
    n2 = wordcount[1]
    for k in range(len(wk)):
        k1 = d1[wk[k]]
        k2 = d2[wk[k]]
        if k1 * k2 < 59:
            continue
        z, pv = sm.stats.proportions_ztest([k1, k2], [n1, n2])
        
        if pv < 0.2:
            twogram_words[(wk[k][0], wk[k][1])] = (d1[wk[k]], d2[wk[k]], pv)
            










def process(s):
    global wordpat
    global sentiment_words
    global twogram_words


    tmpdict = defaultdict(lambda: 0)
    xylist = []
    nwords = 0
    for w in wordpat.finditer(s):
        x,y = w.span()
        spell = SpellChecker()
        key = spell.correction(s[x:y].lower())
        #print(key)
        key = wordnet_lemmatizer.lemmatize(key, pos="v")
        if key in sentiment_words:
            
            tmpdict[key] += 1
            nwords += 1
    print(tmpdict)
    pos_score = 1.0
    neg_score = 1.0

    slist = []
    list_of_subsets = list(map(list, itertools.combinations(tmpdict.keys(), 2)))

    print(list_of_subsets)
    for key in list_of_subsets:
        pos_f = 1
        neg_f = 1
        factors = sentiment_words[key[1]]
        if (key[0], key[1]) in twogram_words:
            #print(twogram_words[(key[0], key[1])])
            pos_bifactors = twogram_words[(key[0], key[1])][0] / (twogram_words[(key[0], key[1])][0] + twogram_words[(key[0], key[1])][1])
            neg_bifactors = twogram_words[(key[0], key[1])][1] / (
                        twogram_words[(key[0], key[1])][0] + twogram_words[(key[0], key[1])][1])
            #print(factors)
            #print(bifactors)
            pos_f = pos_bifactors / factors[0]
            neg_f = neg_bifactors / factors[1]
        else:
            pos_bifactors = 1/ len(list_of_subsets)
            neg_bifactors = 1/ len(list_of_subsets)

            pos_f = pos_bifactors / factors[0]
            neg_f = neg_bifactors / factors[1]

        pos_score += math.log(pos_f,10)
        neg_score += math.log(neg_f, 10)

        print(pos_score)
        print(neg_score)
        print(pos_f)
        print(neg_f)

        
    return pos_score,neg_score , xylist, slist

if __name__ == "__main__":

    for w in stopwords:
        stopdict[w] = True

    pathlist = ["positive.xml", "negative.xml"]

    sentiment_word(pathlist)
    process_2gram(pathlist)

    path = "unlabeled.xml"

    doctree = ET.parse(path)
    root = doctree.getroot()
   

    for child in root:
        for item in child:
            if item.tag == "product_name":
                print(item.text)
            if item.tag == "review_text":
                p_score, n_score, xy, slist = process(item.text)

                

                for t in slist:
                    print("{0:7.3f}  {1:s}".format(t[1], t[0]))
                if p_score >= n_score:
                    guess = "positive"
                    score = p_score
                else :
                    guess = "negative"
                    score= n_score
                print("score = {0:0.6f} which is {1:s}".format(score, guess))
        reply = input("? ")
        
