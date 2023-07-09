#!/usr/bin/env python
# coding: utf-8

# #                                           MAHABHARAT

# ![image.png](attachment:image.png)

# ### Description :

# It narrates the struggle between two groups of cousins in the Kurukshetra War and the fates of the Kaurava and the Pāṇḍava princes and their successors.
# 
# It also contains philosophical and devotional material, such as a discussion of the four "goals of life" or puruṣārtha. Among the principal works and stories in the Mahābhārata are the Bhagavad Gita, the story of Damayanti, the story of Shakuntala, the story of Pururava and Urvashi, the story of Savitri and Satyavan, the story of Kacha and Devayani, the story of Rishyasringa and an abbreviated version of the Rāmāyaṇa, often considered as works in their own right.
# 
# 
# Krishna and Arjuna at Kurukshetra, 18th–19th-century painting.
# Traditionally, the authorship of the Mahābhārata is attributed to Vyāsa. There have been many attempts to unravel its historical growth and compositional layers. The bulk of the Mahābhārata was probably compiled between the 3rd century BCE and the 3rd century CE, with the oldest preserved parts not much older than around 400 BCE. The text probably reached its final form by the early Gupta period (c. 4th century CE).
# 
# The Mahābhārata is the longest epic poem known and has been described as "the longest poem ever written".

# # STEP 1 : Installing the nltk and Downloading the data

# We will use the NLTK package in Python. In this step we will install NLTK and open and jread the text text file

# ![image.png](attachment:image.png)

# In[1]:


# for open the text file
text_file = open("C:/Users/ankit/OneDrive/Documents/1-18 books combined.txt")


# In[2]:


text = text_file.read()


# In[3]:


print(type(text))
print("\n")


# In[4]:


print(text)
print("\n")


# # STEP 2 : Tokenize the data

# Language in its original form cannot be accurately processed by a machine, so you need to process the language to make it easier for the machine to understand. The first part of making sense of the data is through a process called tokenization, or splitting strings into smaller parts called tokens.
# 
# A token is a sequence of characters in text that serves as a unit. Based on how you create the tokens, they may consist of words, emoticons, hashtags, links, or even individual characters. A basic way of breaking language into tokens is by splitting the text based on whitespace and punctuation.

# ![image.png](attachment:image.png)

# In[5]:


#pip install nltk


# In[6]:


#importing important libraries such as sentence tokenize and word tokenize
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize


# In[7]:


#tokenize the text by sentences :
sentences = sent_tokenize(text)


# In[8]:


#tokenize the text by words:
words = word_tokenize(text)


# In[9]:


#how many words are there
print(len(words))


# In[10]:


# print the words:
print(words)


# In[11]:


#print sentences :
print(sentences)


# In[12]:


#import required libraries
from nltk.probability import FreqDist


# In[13]:


#Find the frequency
fdist= FreqDist(words)


# In[14]:


#print 10 most common words :
fdist.most_common(10)


# In[15]:


sentences


# In[16]:


words


# In[17]:


pip install matplotlib


# In[83]:


#plot the graph for fdist:
import matplotlib.pyplot as plt
fdist.plot(10)
fig = plt.figure(figsize=(1,2))


# # STEP 3 : Normalizing the Data

# Normalization helps group together words with the same meaning but different forms. Without normalization, “ran”, “runs”, and “running” would be treated as different words, even though you may want them to be treated as the same word.
# 
# Stemming is a process of removing affixes from a word. Stemming, working with only simple verb forms, is a heuristic process that removes the ends of words.
# 
# The lemmatization algorithm analyzes the structure of the word and its context to convert it to a normalized form. Therefore, it comes at a cost of speed. A comparison of stemming and lemmatization ultimately comes down to a trade off between speed and accuracy.

# ![image.png](attachment:image.png)

# ### Lemmatization 

# In[19]:


import nltk
from nltk.stem import WordNetLemmatizer #for lemmatization
from nltk.tokenize import word_tokenize


# In[20]:


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

# In[21]:


import nltk
from nltk.stem import PorterStemmer #nltk module for stemming
from nltk.tokenize import word_tokenize #tokenizing the text into words


# In[22]:


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


# ### POS Tagging

# In[23]:


# pos tagging of words
tags = nltk.pos_tag(words)
tags


# ### Bags of Words 

# In[24]:


pip install -U scikit-learn


# In[25]:


#importing important library countvectorizer from sklearn
from sklearn.feature_extraction.text import CountVectorizer


# In[26]:


# create an object
cv = CountVectorizer()


# In[27]:


# sklearn is a library it's also known as scikiplearn
# countvector is a tool for extracting text,convert a collection of text document into matrix of token count


# In[28]:


# Generate output for bag of words
B_O_W = cv.fit_transform(sentences).toarray()


# In[29]:


# total words with index in model:
cv.vocabulary_


# In[30]:


# show the output
B_O_W


# # STEP 4 : Removing Noise from the Data

# In this step, we will remove noise from the dataset. Noise is any part of the text that does not add meaning or information to data.
# 
# Noise is specific to each project, so what constitutes noise in one project may not be in a different project. For instance, the most common words in a language are called stop words. Some examples of stop words are “is”, “the”, and “a”. They are generally irrelevant when processing language, unless a specific use case warrants their inclusion.

# ![image.png](attachment:image.png)

# ### Removing Punctuation

# In[31]:


# empty list to share words
words_no_punc = []


# In[32]:


for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())


# In[33]:


print(words_no_punc)
print("\n")


# In[34]:


#printing the length of words after removing punctuation
print(len(words_no_punc))


# In[35]:


fdist = FreqDist(words_no_punc)
fdist.most_common(10)


# In[36]:


fdist.plot(10)


# ### Removing Stopwords

# In[37]:


#importing important library stopwords from nltk
from nltk.corpus import stopwords


# In[38]:


#list of stopwords
stopwords = stopwords.words("english")


# In[39]:


#printing after removing stopwords
print(stopwords)


# In[40]:


#empty list to store clean words
clean_words = []


# In[41]:


# append used to add new element(words, sent, number etc)at the end of the list


# In[42]:


for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)


# In[43]:


print(clean_words)#to print all clean words
print("\n")#to add empty line for better understanding
print(len(clean_words))#to check the length of clean words


# In[44]:


#final frequencing distribution
fdist = FreqDist(clean_words)
fdist.most_common(10)


# In[45]:


fdist.plot(10)


# In[46]:


from nltk import sent_tokenize, PorterStemmer, word_tokenize
from nltk.corpus import stopwords


# In[47]:


# for removing stopwords from the text file
sw = set(stopwords.words('english'))
print("================================")
print(sw)
print("================================")
print(len(sw))
print("================================")


# # STEP 5 : Creating Wordcloud  

# Word Cloud provides an excellent option to analyze the text data through visualization in the form of tags, or words, where the importance of a word is explained by its frequency.
# It's important to think about the context of the text and the specific meanings of the words being used. Be wary of outliers: Sometimes, a word may appear very large in the word cloud simply because it appears frequently, even if it's not particularly meaningful or relevant to the overall message of the text.

# ![image.png](attachment:image.png)

# In[48]:


#pip install wordcloud


# In[49]:


# importing important library for word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# In[50]:


# import/open the image in jpg formt(copy the path)
im = np.array(Image.open("C:/Users/ankit/OneDrive/Desktop/mahabarat 4.jpeg"))


# In[51]:


# mask image is that defines the shape of the wordcloud
wordcloud = WordCloud(mask = im).generate(text)
plt.figure(figsize = (7, 5))
plt.imshow(wordcloud)
plt.show()


# ###  Positive and Negative word cloud 

# In[52]:


#The VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon is a pre-built lexicon (or dictionary) of 
#sentiment-related words and phrases in English.(positive & negative)


# In[53]:


import nltk
nltk.download('vader_lexicon')


# In[54]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create an instance of the SentimentIntensityAnalyzer class
sentiment_analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis using the SentimentIntensityAnalyzer object
sentiments = [(word, sentiment_analyzer.polarity_scores(word)['compound']) for word in words]


# In[55]:


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


# In[56]:


# to specify the height, font, colour of the word cloud
negative_wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stop_words, 
                min_font_size = 10).generate(" ".join(negative_words))

negative_wordcloud


# In[57]:


#creating positive words
plt.figure(figsize=(3, 3), facecolor=None)
plt.imshow(positive_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[58]:


#creating negative words
plt.figure(figsize=(3, 3), facecolor=None)
plt.imshow(negative_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# ### Translation 

# In[59]:


pip install googletrans==4.0.0-rc1


# In[60]:


from googletrans import Translator

translator = Translator()
sentence ="I am the compiler of Vedanta and, indeed, I am the knower of the Vedas."
            
translated = translator.translate(sentence, src='en', dest='de')

print(translated.text)


# In[61]:


translator = Translator()
sentence ="I am the compiler of Vedanta and, indeed, I am the knower of the Vedas."
            
translated = translator.translate(sentence, src='en', dest='ka')

print(translated.text)


# # STEP 6 : Summarizing

# Text summarization in NLP is the process of summarizing the information in large texts for quicker consumption.

# ![image.png](attachment:image.png)

# ###  Creating Frequency Matrix for the tokens 

# In[62]:


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

# In[63]:


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

# In[64]:


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

# In[65]:


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

# In[66]:


def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix


# ### Weighing the words in a sentence - Scoring

# In[67]:


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


# ###  Average sentence score - Threshold

# In[68]:


def find_average_score(sentence_val) -> int:
    sum_values = 0
    for entry in sentence_val:
        sum_values = sum_values + sentence_val[entry]
    average = sum_values / len(sentence_val)
    return average


# ### Call everything and get the summarization done

# In[69]:


def generate_summary(sentences, sentence_val, threshold):
    sentence_count = 0
    summary = ""
    for sentence in sentences:
        if sentence[:15] in sentence_val and sentence_val[sentence[:15]] >= (threshold):
            summary = summary + " " + sentence
            sentence_count = sentence_count + 1
    return summary


# In[70]:


freq_matrix = create_frequency_matrix(sentences)


# In[71]:


tf_matrix = create_tf_matrix(freq_matrix)


# In[72]:


count_doc_per_words = create_document_per_words(freq_matrix)


# In[73]:


# Define and assign a value to total_documents
total_documents = 1000
# Call create_idf_matrix function with freq_matrix, count_doc_per_words, and total_documents
idf_matrix = create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)


# In[74]:


tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)


# In[75]:


sentence_scores = score_sentences(tf_idf_matrix)


# In[76]:


threshold = find_average_score(sentence_scores)


# In[77]:


summary = generate_summary(sentences, sentence_scores, threshold)
summary


# ### Conclusion : 

# In this project, first we import, open, read & print the script of joker movie in a text form to begin with, first we check the type of data this files contain i.e.,string type,and the length of the file. Next step is to tokenize the file on the bases of words and sentences by importing important libraries such as sent_tokenize & word_tokenize from nltk package,Now let's find the frequency  of the most common words in the file, from which, I have imported the FreqDist function. With the help of matplotlib,I have created the graph of the total words & sentences. After this remove unnecessary words,i.e removing punctuation and make the plot from it. To increase the search performance, I have removed stopwords i.e remove the low-level information from our text in order to give more focus to the important information.Next step is to be done with Stemming and Lemmatization, so that it reduced those words which are not important and make our data easy to work with. Now, I will categorize the words in a text (corpus) in correspondence with a particular part of speech, i.e POS tagging and analyse text document based on word count (Bag of Words).Now to summarize the data I will be creating frequency Matrix for the tokens. Making of word cloud on the basis of sentiment analysis of words having positive, negative or netural meaning. For that I have imported few images in which shape I want to make my word cloud.I wanted to check how the data of each paragraph is interrelated with each other, and on the basis of that data, I have done clustering using scatterd plot. This will show which data belongs to which cluster and how they are correspondence with each other. For clustering, we need to import libraries like pandas, numpy, matplotlib, kmeans, TfidfVectorizer.To get the data in any other language we can translate the data into that language. To check this I have translate one sentence into Korean and Hindi, for which I have to install Googletrans library and from it importing the translator. 

# ![image-2.png](attachment:image-2.png)

# In[ ]:




