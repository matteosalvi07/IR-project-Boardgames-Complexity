# Complexity in Boardgames

#### In this project my aim is to find metric for the ranking of Boardgames based on complexity as project of Information Retrieval exam.
## Class definitions


```python
class Rulebook:
    def __init__(self, filename, text, pagenum,num_sent,num_word,num_complex_word,syn_analysis):
        self.filename = filename
        self.text = text
        self.pagenum= pagenum
        self.num_word=num_word
        self.num_sent=num_sent
        self.num_complex_word=num_complex_word
        self.readability=0.4*(self.num_word/self.num_sent + 100*self.num_complex_word/self.num_word) #readability metric
        self.syn_analysis=syn_analysis
```


```python
class BoardGame:

    def __init__(self, name, comments,weight,expansions_num,rulebook,score):
        self.name = name
        self.comments = comments
        self.BGGweight=weight
        self.expansions_num = expansions_num
        self.rulebook=rulebook
        self.comment_score=score
        self.total_weight=0

```

## Usefull methods


```python
from spacy_syllables import SpacySyllables
import spacy_fastlang
import spacy
import pandas as pd
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
import requests
import xml.etree.ElementTree as et
from os import listdir
from os.path import isfile, join
from string import punctuation
import nltk
from collections import defaultdict
```


```python
sw = stopwords.words('english')
bggxmlapi2 = 'https://boardgamegeek.com/xmlapi2/'
bggxmlapi = 'http://www.boardgamegeek.com/xmlapi/'


mypath = './Rulebook/'
easy= "accessible,clear,effortless,obvious,painless,simple,smooth,straightforward,uncomplicated,apparent,basic,child's play,cinch,easily done,elementary,evident,facile,inconsiderable,light,little,manageable,manifest,mere,no bother,no problem,no sweat,no trouble,not burdensome,nothing to it,paltry,picnic,piece of cake,plain,plain sailing,pushover,royal,simple as ABC,slight,snap,undemanding,uninvolved,untroublesome,wieldy,yielding,easy,"
difficult="difficult,ambitious,arduous,burdensome,challenging,crucial,demanding,laborious,onerous,painful,problematic,severe,strenuous,tough,troublesome,backbreaker,bothersome,difficile,easier said than done,effortful,exacting,formidable,galling,gargantuan,hard-won,heavy,herculean,immense,intricate,irritating,labored,no picnic,not easy,operose,problem,prohibitive,rigid,stiff,titanic,toilsome,trying,unyielding,uphill,upstream,wearisome,"
difficult_word=[option.strip() for option in difficult.split(',')]
easy_word=[option.strip() for option in easy.split(',')]


#loaded the model for nlp with spacy and the pipelines
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("syllables", after="tagger")
nlp.add_pipe("language_detector")
```





```python
def read_files(): #to read file in a directory
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

```


```python
def read_rolebook_pdf(file): #read the content of a pdf to create an istance of a rulebook
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
    return Rulebook(filename, text, reader.getNumPages(), num_sent, num_word, num_complex_word, syn_analysis)

```


```python
def process_text(corpus): #process rulebook content to the sintactic analysis
    tokens = []
    for sentence in corpus:
        for token in sentence:
            tokens.append({
                'position': token.idx, 'text': token.text, 'pos': token.pos_, 'lemma': token.lemma_,
                'alpha': token.is_alpha, 'stop': token.is_stop,
                'morph': token.morph, 'syll_count': token._.syllables_count
            })
    return pd.DataFrame(tokens)

```


```python
def synt_analysis(pos): #metric from syntactic analysis
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


```


```python

def create_boardgame(rulebook): #use bgg api's to download data of each boardgame from the rulebook
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
    eng_comments=select_english_comments(comments)
    score=analyze_comments(eng_comments)

    return BoardGame(name.text, eng_comments, weight.text, len(expansions), rulebook,score)

```


```python

def select_english_comments(comments): #to select only comments in english
    eng_comm=[]
    for c in comments:
        doc = nlp(c)
        eng_comm.append({"comm": c, "lang": doc._.language})
    return [x["comm"] for x in eng_comm if x["lang"]=="en" ]



```


```python

def analyze_comments(comments): #creation of the metric from users' comments
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


```

## Let's calculate


```python
files = read_files()
boardgames=[]

for file in files:
    rulebook=read_rolebook_pdf(file)
    boardgame=create_boardgame(rulebook)
    boardgames.append(boardgame)


for b in boardgames:
    b.total_weight=b.rulebook.readability+ b.rulebook.syn_analysis+b.comment_score
    if b.expansions_num>6:
        b.total_weight=b.total_weight+1
    print(b.name, b.BGGweight, b.total_weight)


```

    CATAN 2.3032 6.415468242213069
    Die Macher 4.3144 9.141517262128437
    Samurai 2.4794 6.562581847635009
    
