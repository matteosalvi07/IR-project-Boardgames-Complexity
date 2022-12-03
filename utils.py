from spacy_syllables import SpacySyllables
import spacy
import pandas as pd
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
import model.rulebook as rb
import requests
import xml.etree.ElementTree as et
import model.boardgame as bg
from os import listdir
from os.path import isfile, join
import helper
from string import punctuation
import nltk
from collections import defaultdict

config = helper.read_config()

sw = stopwords.words('english')

mypath = config['Directory']['mypath']

bggxmlapi2 = config['API']['bggxmlapi2']
bggxmlapi = config['API']['bggxmlapi']

difficult_word=[option.strip() for option in config.get('WORDS', 'difficult_word').split(',')]
easy_word=[option.strip() for option in config.get('WORDS', 'easy_word').split(',')]



nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("syllables", after="tagger")


def read_files():
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def process_text(corpus):
    tokens = []
    for sentence in corpus:
        for token in sentence:
            tokens.append({
                'position': token.idx, 'text': token.text, 'pos': token.pos_, 'lemma': token.lemma_,
                'alpha': token.is_alpha, 'stop': token.is_stop,
                'morph': token.morph, 'syll_count': token._.syllables_count
            })
    return pd.DataFrame(tokens)


def synt_analysis(pos):
    aggettivi = pos['ADJ'] + pos['NUM']
    nomi = pos['NOUN'] + pos['PROPN']
    pronomi = pos['PRON']
    spazi = pos['SPACE']
    verbi = pos['VERB']
    coord_sub = pos['CCONJ'] + pos['SCONJ']
    app = pos['ADP']
    avv = pos['ADV']
    determ = pos['DET']
    st1 = nomi / verbi
    st2 = coord_sub / verbi
    st3 = app / verbi
    st4 = avv / verbi
    st5 = pronomi / nomi
    st6 = determ / nomi
    st7 = aggettivi / nomi
    st8 = spazi / verbi
    met = st1 + 2 * st2 + st3 + st4 + st5 * 0.8 + st6 * 0.8 + st7 * 0.5 - 0.7 * st8
    return met


def read_rolebook_pdf(file):
    reader = PdfReader(mypath + file)
    filename = file
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    corpus = [sentence for sentence in nlp(text).sents]
    df = process_text(corpus)
    num_sent = len(text)
    num_word = df.query('alpha==True').shape[0]
    num_complex_word = df.query('alpha==True&syll_count>=3').shape[0]
    pos = df.groupby(["pos"]).size()
    syn_analysis = synt_analysis(pos)
    return rb.Rulebook(filename, text, reader.getNumPages(), num_sent, num_word, num_complex_word, syn_analysis)


def create_boardgame(rulebook):
    req_for_id = bggxmlapi2 + "search?query=" + rulebook.filename.split(".")[0] + "&type=boardgame&exact=1"
    res_for_id = requests.get(req_for_id)
    root_for_id = et.fromstring(res_for_id.text)
    game_id = root_for_id[0].get('id')
    req_for_game = bggxmlapi + "boardgame/" + game_id
    res_for_game = requests.get(req_for_game + "?comments=1&stats=1")
    game = et.fromstring(res_for_game.text)[0]
    name = game.find("name[@primary='true']")
    weight = game.find("./statistics/ratings/averageweight")
    expansions = game.findall("boardgameexpansion")
    comments = []

    for comment in game.findall("./comment"):
        comments.append(comment.text)

    score=analyze_comments(comments)

    return bg.BoardGame(name.text, comments, weight.text, len(expansions), rulebook,score)






def analyze_comments(comments):
    I = defaultdict(set)
    nltk_tokenize = lambda text: [x.lower() for x in nltk.word_tokenize(text) if x not in punctuation]
    corpus = list(enumerate(comments))
    for i, document in corpus:
        tokens = nltk_tokenize(document.lower())
        for token in tokens:
            I[token].add(i)

    easy_word_occurrence=0
    for word in easy_word:
        easy_word_occurrence += len(I[word])

    difficult_word_occurrence = 0
    for word in difficult_word:
        difficult_word_occurrence += len(I[word])

    B = defaultdict(lambda: 0)

    for doc in comments:
        for a, b in nltk.ngrams(nltk_tokenize(doc), 2):
            B[(a, b)] += 1

    difficult_word_occurrence_neg = 0
    for word in difficult_word:
        difficult_word_occurrence_neg += B[("not", word)]

    easy_word_occurrence_neg = 0
    for word in easy_word:
        easy_word_occurrence_neg += B[("not", word)]

    easy_word_occurrence_tot=easy_word_occurrence-easy_word_occurrence_neg
    difficult_word_occurrence_tot=difficult_word_occurrence-difficult_word_occurrence_neg

    if easy_word_occurrence_tot/difficult_word_occurrence_tot > 1.5:
        score=-2
    elif difficult_word_occurrence_tot/easy_word_occurrence_tot > 1.5:
        score=1
    else:
        score=0

    return score

