pip install nltk
import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

text = "Text mining is also referred to as text analytics. Text mining is a process of exploring sizable textual data and finding patterns. Text Mining processes the text itself, while NLP processes with the underlying metadata."

from nltk.tokenize import sent_tokenize
tokenized_text = sent_tokenize(text)
print("Sentence Tokenization")
print("--------------------")
print(tokenized_text)

from nltk.tokenize import word_tokenize
tokenized_word = word_tokenize(text)
print("Word Tokenization")
print("--------------------")
print(tokenized_word)

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
print("Stop Words in English")
print("--------------------")
print(stop_words)

import re
text = re.sub('[^a-zA-Z]', ' ', text)
tokens = word_tokenize(text.lower())
filtered_text = []
for w in tokens:
    if w not in stop_words:
        filtered_text.append(w)
print("Tokenized Sentence:", tokens)
print("Filtered Sentence:", filtered_text)

from nltk.stem import PorterStemmer
e_words = ["wait", "waiting", "waited", "waits"]
ps = PorterStemmer()
for w in e_words:
    rootWord = ps.stem(w)
print(rootWord)

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))

pos_tags = nltk.pos_tag(filtered_text)
print("POS Tagging:")
print(pos_tags)

documentA = "Jupiter is the largest Planet"
documentB = "Mars is the fourth planet from the Sun"

def computeTF(wordDict, bagOfWords):
tfDict = {}
bagOfWordsCount = len(bagOfWords)
for word, count in wordDict.items():
tfDict[word] = count / float(bagOfWordsCount)
return tfDict

# Calculate inverse document frequency

def computeIDF(documents):
N = len(documents)
idfDict = dict.fromkeys(documents[0].keys(), 0)
for document in documents:
for word, val in document.items():
if val > 0:
idfDict[word] += 1
for word, val in idfDict.items():
idfDict[word] = math.log(N / float(val))
return idfDict

# Calculate TF-IDF

def computeTFIDF(tfBagOfWords, idfs):
tfidf = {}
for word, val in tfBagOfWords.items():
tfidf[word] = val \* idfs[word]
return tfidf

# Create DataFrames to show results

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
df = pd.DataFrame([tfidfA, tfidfB])
df

