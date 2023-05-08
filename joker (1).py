#!/usr/bin/env python
# coding: utf-8

# #                                              JOKER SCRIPT

# ![image-5.png](attachment:image-5.png)

# ### Discription

# Joker is a 2019 American psychological thriller film directed and produced by Todd Phillips, who co-wrote the screenplay 
# with Scott Silver. The film, based on DC Comics characters, stars Joaquin Phoenix as the Joker.

# ### About NLP: 

# Natural Language Processing is the practice of teaching machines to understand and interpret conversational inputs from humans. NLP based on Machine Learning can be used to establish communication channels between humans and machines.The different implementations of NLP can help businesses and individuals save time, improve efficiency and increase customer satisfaction.
# 
# Sentiment analysis uses NLP and ML to interpret and analyze emotions in subjective data like news articles and tweets. Positive, negative, and neutral opinions can be identified to determine a customer’s sentiment towards a brand, product, or service. Sentiment analysis is used to gauge public opinion, monitor brand reputation, and better understand customer experiences.

# In[10]:


# for open the text file
text_file = open("C:/Users/ankit/Downloads/joker_script.txt")


# In[11]:


# for reading the text file
text = text_file.read()


# In[12]:


# for knowing the data type of the text file
print(type(text))
print("\n")


# In[13]:


# for viewing the text file
print(text)
print("\n")


# In[14]:


# for the length of the text of the file
print(len(text))


# In[15]:


#pip install nltk


# In[16]:


#importing important libraries such as sentence tokenize and word tokenize
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize


# In[17]:


#tokenize the text by sentences :
sentences = sent_tokenize(text)


# In[18]:


#tokenize the text by words:
words = word_tokenize(text)


# In[19]:


#how many words are there
print(len(words))


# In[20]:


# print the words:
print(words)


# In[21]:


#print sentences :
print(sentences)


# In[22]:


#import required libraries
from nltk.probability import FreqDist


# In[23]:


#Find the frequency
fdist= FreqDist(words)


# In[24]:


#print 10 most common words :
fdist.most_common(10)


# In[25]:


sentences


# In[26]:


words


# In[27]:


pip install matplotlib


# In[28]:


#plot the graph for fdist:
import matplotlib.pyplot as plt
fdist.plot(10)
fig = plt.figure(figsize=(1, 1))


# ## Removing Punctuation

# In[29]:


# empty list to share words
words_no_punc = []


# In[30]:


for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())


# In[31]:


print(words_no_punc)
print("\n")


# In[32]:


#printing the length of words after removing punctuation
print(len(words_no_punc))


# In[33]:


fdist = FreqDist(words_no_punc)
fdist.most_common(10)


# In[34]:


fdist.plot(10)


# ### Removing Stopwords

# In[35]:


#importing important library stopwords from nltk
from nltk.corpus import stopwords


# In[36]:


#list of stopwords
stopwords = stopwords.words("english")


# In[37]:


#printing after removing stopwords
print(stopwords)


# In[38]:


#empty list to store clean words
clean_words = []


# In[39]:


# append used to add new element(words, sent, number etc)at the end of the list


# In[40]:


for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)


# In[41]:


print(clean_words)#to print all clean words
print("\n")#to add empty line for better understanding
print(len(clean_words))#to check the length of clean words


# In[42]:


#final frequencing distribution
fdist = FreqDist(clean_words)
fdist.most_common(10)


# In[43]:


fdist.plot(10)


# In[44]:


from nltk import sent_tokenize, PorterStemmer, word_tokenize
from nltk.corpus import stopwords


# In[45]:


# for removing stopwords from the text file
sw = set(stopwords.words('english'))
print("================================")
print(sw)
print("================================")
print(len(sw))
print("================================")


# ### Lemmatization 

# In[46]:


import nltk
from nltk.stem import WordNetLemmatizer #for lemmatization
from nltk.tokenize import word_tokenize


# In[47]:


# initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

words = word_tokenize(text)
# lemmatize each word in the text
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

# join the lemmatized words into a string
lemmatized_text = ' '.join(lemmatized_words)

# print the lemmatized text
print(lemmatized_text)


# ### Stemming

# In[48]:


import nltk
from nltk.stem import PorterStemmer #nltk module for stemming
from nltk.tokenize import word_tokenize #tokenizing the text into words


# In[49]:


# initialize the stemmer
stemmer = PorterStemmer()

# tokenize the text into words
words = word_tokenize(text)
    # stem each word in the text
stemmed_words = [stemmer.stem(word) for word in words]

# join the stemmed words into a string
stemmed_text = ' '.join(stemmed_words)

# print the stemmed text
print(stemmed_text)


# ### POS tagging

# In[50]:


# pos tagging of words
tags = nltk.pos_tag(words)
tags


# ### Bag of Words

# In[51]:


pip install -U scikit-learn


# In[52]:


#importing important library countvectorizer from sklearn
from sklearn.feature_extraction.text import CountVectorizer


# In[53]:


# create an object
cv = CountVectorizer()


# In[54]:


# sklearn is a library it's also known as scikiplearn
# countvector is a tool for extracting text,convert a collection of text document into matrix of token count


# In[55]:


# Generate output for bag of words
B_O_W = cv.fit_transform(sentences).toarray()


# In[56]:


# total words with index in model:
cv.vocabulary_


# In[57]:


# features:
cv.get_feature_names()


# In[58]:


# show the output
B_O_W


# ### Creating Wordcloud

# In[59]:


#pip install wordcloud


# In[60]:


# importing important library for word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# In[61]:


# import/open the image in jpg formt(copy the path)
im = np.array(Image.open("C:/Users/ankit/OneDrive/Desktop/joker.jpg"))


# In[62]:


# mask image is that defines the shape of the wordcloud
wordcloud = WordCloud(mask = im).generate(text)
plt.figure(figsize = (3, 4))
plt.imshow(wordcloud)
plt.show()


# ## Positive and Negative word cloud

# In[63]:


#The VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon is a pre-built lexicon (or dictionary) of 
#sentiment-related words and phrases in English.(positive & negative)


# In[64]:


import nltk
nltk.download('vader_lexicon')


# In[65]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create an instance of the SentimentIntensityAnalyzer class
sentiment_analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis using the SentimentIntensityAnalyzer object
sentiments = [(word, sentiment_analyzer.polarity_scores(word)['compound']) for word in words]


# In[66]:


#creates two lists of words:(+ -)based on sentiment scores of words calculated using vader
from nltk.corpus import stopwords
from wordcloud import WordCloud

positive_words = [word for word, sentiment in sentiments if sentiment > 0]
negative_words = [word for word, sentiment in sentiments if sentiment < 0]

stop_words = set(stopwords.words("english")) # Define stop_words

positive_wordcloud = WordCloud(width=800, height=800,
                               background_color='white',
                               stopwords=stop_words,
                               min_font_size=10).generate(" ".join(positive_words))

positive_wordcloud


# In[67]:


# to specify the height, font, colour of the word cloud
negative_wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stop_words, 
                min_font_size = 10).generate(" ".join(negative_words))

negative_wordcloud


# In[68]:


#creating positive words
plt.figure(figsize=(3, 3), facecolor=None)
plt.imshow(positive_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[69]:


#creating negative words
plt.figure(figsize=(3, 3), facecolor=None)
plt.imshow(negative_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# ### Translation

# In[70]:


pip install googletrans==4.0.0-rc1


# In[71]:


from googletrans import Translator

translator = Translator()
sentence ="The social worker gives him a look, then reads something"
            
translated = translator.translate(sentence, src='en', dest='ko')

print(translated.text)


# In[72]:


from googletrans import Translator

translator = Translator()
sentence ="The social worker gives him a look, then reads something"
translated = translator.translate(sentence, src='en', dest='hindi')

print(translated.text)


# ### Clustering

# In[80]:


pip install pandas


# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Define the documents
docs = [
    "I used to think that my life was a tragedy, but now I realize, it's a comedy.",
    "I hope my death makes more cents than my life.",
    "Is it just me, or is it getting crazier out there?",
    "The worst part of having a mental illness is people expect you to behave as if you don't."
]

# Convert the documents into a matrix of TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(docs)

# Perform k-means clustering with k=2
k = 2
model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

# Reduce the dimensionality of the feature matrix to 2 dimensions for plotting
pca = PCA(n_components=2).fit(X.toarray())
data2D = pca.transform(X.toarray())

# Plot the clusters
plt.figure(figsize=(5, 6))
for i in range(k):
    points = np.array([data2D[j] for j in range(len(docs)) if model.labels_[j] == i])
    plt.scatter(points[:, 0], points[:, 1], s=30, label=f'Cluster {i+1}')
plt.legend()
plt.title(f'K-means clustering of {len(docs)} documents')
plt.show()


# ###  Creating Frequency Matrix for the tokens

# In[82]:


def create_frequency_matrix(sentences):
    frequency_matrix = {}
    sw = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    for sent in sentences: 
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in sw:
                continue
            if word in freq_table:
                freq_table[word] = freq_table[word] + 1
            else:
                freq_table[word] = 1
        frequency_matrix[sent[:15]] = freq_table
    return frequency_matrix


# ###  Manual Term-Frequency Computation

# In[83]:


def create_tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, f_table in freq_matrix.items():
        tf_table = {}
        count_words_in_sent = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sent
        tf_matrix[sent] = tf_table
    return tf_matrix


# ### Creating a table for document per words

# In[84]:


def create_document_per_words(freq_matrix):
    word_per_doc_table = {}
    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] = word_per_doc_table[word] + 1
            else:
                word_per_doc_table[word] = 1
    return word_per_doc_table


# ### TF-IDF Computation

# In[85]:


import math
def create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))
        idf_matrix[sent] = idf_table
    return idf_matrix


# ### TF-IDF Computation

# In[86]:


def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix


# ## Weighing the words in a sentence - Scoring

# In[87]:


def score_sentences(tf_idf_matrix) -> dict: 
    sentence_val = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence = total_score_per_sentence + score
        if count_words_in_sentence > 0:
            sentence_val[sent] = total_score_per_sentence / count_words_in_sentence
    return sentence_val


# ## Average sentence score - Threshold

# In[88]:


def find_average_score(sentence_val) -> int:
    sum_values = 0
    for entry in sentence_val:
        sum_values = sum_values + sentence_val[entry]
    average = sum_values / len(sentence_val)
    return average


# ## Call everything and get the summarization done

# In[89]:


def generate_summary(sentences, sentence_val, threshold):
    sentence_count = 0
    summary = ""
    for sentence in sentences:
        if sentence[:15] in sentence_val and sentence_val[sentence[:15]] >= (threshold):
            summary = summary + " " + sentence
            sentence_count = sentence_count + 1
    return summary


# In[90]:


freq_matrix = create_frequency_matrix(sentences)


# In[91]:


tf_matrix = create_tf_matrix(freq_matrix)


# In[92]:


count_doc_per_words = create_document_per_words(freq_matrix)


# In[93]:


# Define and assign a value to total_documents
total_documents = 1000
# Call create_idf_matrix function with freq_matrix, count_doc_per_words, and total_documents
idf_matrix = create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)


# In[94]:


tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)


# In[95]:


sentence_scores = score_sentences(tf_idf_matrix)


# In[96]:


threshold = find_average_score(sentence_scores)


# In[97]:


summary = generate_summary(sentences, sentence_scores, threshold)
summary


# ### Conclusion:

# In this project, first we import, open, read & print the script of joker movie in a text form to begin with, first we check the type of data this files contain i.e.,string type,and the length of the file. Next step is to tokenize the file on the bases of words and sentences by importing important libraries such as sent_tokenize & word_tokenize from nltk package,Now let's find the frequency  of the most common words in the file, from which, I have imported the FreqDist function. With the help of matplotlib,I have created the graph of the total words & sentences. After this remove unnecessary words,i.e removing punctuation and make the plot from it. To increase the search performance, I have removed stopwords i.e remove the low-level information from our text in order to give more focus to the important information.Next step is to be done with Stemming and Lemmatization, so that it reduced those words which are not important and make our data easy to work with. Now, I will categorize the words in a text (corpus) in correspondence with a particular part of speech, i.e POS tagging and analyse text document based on word count (Bag of Words).Now to summarize the data I will be creating frequency Matrix for the tokens. Making of word cloud on the basis of sentiment analysis of words having positive, negative or netural meaning. For that I have imported few images in which shape I want to make my word cloud.I wanted to check how the data of each paragraph is interrelated with each other, and on the basis of that data, I have done clustering using scatterd plot. This will show which data belongs to which cluster and how they are correspondence with each other. For clustering, we need to import libraries like pandas, numpy, matplotlib, kmeans, TfidfVectorizer.To get the data in any other language we can translate the data into that language. To check this I have translate one sentence into Korean and Hindi, for which I have to install Googletrans library and from it importing the translator.    
# 

# In[ ]:




