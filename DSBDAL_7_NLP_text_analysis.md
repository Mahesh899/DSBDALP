# Text Mining and NLP Techniques

This notebook demonstrates various text mining and Natural Language Processing (NLP) techniques using Python's NLTK library.

## Installation and Setup

```python
pip install nltk
import nltk
import pandas as pd
```

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

```

## Sample Text

```python
text = "Text mining is also referred to as text analytics. Text mining is a process of exploring sizable textual data and finding patterns. Text Mining processes the text itself, while NLP processes with the underlying metadata."

```

## Tokenization

### Sentence Tokenization

```python
from nltk.tokenize import sent_tokenize
tokenized_text = sent_tokenize(text)
print("Sentence Tokenization")
print("--------------------")
print(tokenized_text)
```

### Word Tokenization

```python
from nltk.tokenize import word_tokenize
tokenized_word = word_tokenize(text)
print("Word Tokenization")
print("--------------------")
print(tokenized_word)
```

## Stop Words

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
print("Stop Words in English")
print("--------------------")
print(stop_words)
```

## Text Cleaning and Stop Word Removal

```python
import re
text = re.sub('[^a-zA-Z]', ' ', text)
tokens = word_tokenize(text.lower())
filtered_text = []
for w in tokens:
    if w not in stop_words:
        filtered_text.append(w)
print("Tokenized Sentence:", tokens)
print("Filtered Sentence:", filtered_text)
```

## Stemming

```python
from nltk.stem import PorterStemmer
e_words = ["wait", "waiting", "waited", "waits"]
ps = PorterStemmer()
for w in e_words:
    rootWord = ps.stem(w)
print(rootWord)
```

## Lemmatization

```python
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))
```

## Part-of-Speech Tagging

```python
pos_tags = nltk.pos_tag(filtered_text)
print("POS Tagging:")
print(pos_tags)
```

## TF-IDF Implementation

```python
documentA = "Jupiter is the largest Planet"
documentB = "Mars is the fourth planet from the Sun"
```

```
# Create bag of words

bagOfWordsA = documentA.split()
bagOfWordsB = documentB.split()
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))
```

# Calculate term frequency

```python

def computeTF(wordDict, bagOfWords):
tfDict = {}
bagOfWordsCount = len(bagOfWords)
for word, count in wordDict.items():
tfDict[word] = count / float(bagOfWordsCount)
return tfDict
```

```python
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
```

```python
# Calculate TF-IDF

def computeTFIDF(tfBagOfWords, idfs):
tfidf = {}
for word, val in tfBagOfWords.items():
tfidf[word] = val \* idfs[word]
return tfidf
```

```python
# Create DataFrames to show results

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
df = pd.DataFrame([tfidfA, tfidfB])
df

```
