import os
import string
import json
import pymongo
import math
from pymongo import MongoClient
from bs4 import BeautifulSoup
from nltk import word_tokenize, defaultdict
from nltk.stem import WordNetLemmatizer

c = MongoClient('localhost', 27017)
db = c['Index-database-part-B']
writer = db.posts

lemmatizer = WordNetLemmatizer()
Inverted_Index = {}

punctuation = string.punctuation
stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
              "as",
              "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
              "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
              "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
              "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
              "his", "how", "how's", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
              "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
              "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
              "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
              "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
              "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
              "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
              "what",
              "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
              "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
              "yourself", "yourselves", "i"]

path = "C:/Users/Tuan/Desktop/CS 121/Assignment 3/Part B/WEBPAGES_RAW"
abs_url_file_path = "C:/Users/Tuan/Desktop/CS 121/Assignment 3/Part B/WEBPAGES_RAW/bookkeeping.json"
# Read the json file in and parse it into dictionaries with
# keys = the path, value = the url
with open(abs_url_file_path) as url_in_json_file:
    url_corpus = json.load(url_in_json_file)

num_doc = len(url_corpus.keys())  # Total number of document


def create_index():
    for file_path in url_corpus.keys():
        abs_file_path = path + "/" + file_path
        # Parse the html file
        soup = BeautifulSoup(open(abs_file_path, encoding="utf-8"), "lxml")
        # Tags that contain texts
        word_tag = soup.find_all(['p', 'b', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'title', 'strong'])
        important_tag = ['b', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'title']
        # token stores the list of tokenized text in the tag
        # temp is a dictionary that stores the token as key and the token_frequency as value
        token = []
        temp = {}
        word_frequency = 1
        # For each tag in word_tag, get the text then tokenize the lower case words
        for tag in word_tag:
            token = word_tokenize(tag.get_text().lower())

            # for each word in token, lemmatize it first, then check if the word is in
            # Inverted_Index or not. If not, add the word and the folder/file number as
            # key/value. If word exists already, then append the folder/file number to
            # that word(key) as another location where the word exists
            for word in token:
                word = lemmatizer.lemmatize(word)
                if word not in stop_words and word.isalnum() and word not in punctuation:
                    if word not in temp:
                        if word in important_tag:
                            temp.update({word: word_frequency + 9})
                        else:
                            temp.update({word: word_frequency})
                    else:
                        word_frequency = temp[word]
                        if word in important_tag:
                            temp[word] = word_frequency + 10
                        else:
                            temp[word] = word_frequency + 1

        # for each key, value in temp:
        #   if the key is not in Inverted_Index
        #       Add that key to Inverted_Index and value as a dict() that contains
        #       file_path as key and word_frequency as the value of that key
        #   else if the key is already in Inverted_Index
        #       Append a dict{file_path: word_frequency} to that key
        for key, value in temp.items():
            if key not in Inverted_Index:
                Inverted_Index.update({key: {file_path: value}})
            else:
                Inverted_Index[key].update({file_path: value})
        print(file_path)

    # For every document that contains the token, calculate tf-idf
    # Upload to database: MongoDB
    for token in Inverted_Index.keys():
        doc_freq = len(Inverted_Index[token].keys())
        for key, value in Inverted_Index[token].items():
            term_freq = value  # term frequency
            Inverted_Index[token][key] = (1 + math.log(term_freq, 10)) * math.log(num_doc / doc_freq, 10)  # tf-idf

        post = {"word": token, "meta_data": Inverted_Index[token]}
        writer.insert_one(post)


score = defaultdict(float)
q_squared = defaultdict(float)
d_squared = defaultdict(float)


def rank_document():
    argv_dict = defaultdict(float)  # argv_dict contains query terms as token and token raw frequencies

    # Get user input and tokenize the query
    # Lemmatize the query then check if the conditions are met
    #  If yes, then increment that token frequency
    user_input = input("Please enter a query: ").lower()
    input_token = word_tokenize(user_input)
    for token in input_token:
        token = lemmatizer.lemmatize(token)
        if token not in stop_words and token.isalnum() and token not in punctuation:
            argv_dict[token] += 1

    # For each token in argv_dict, change the term_freq from raw to log based
    for key, raw_tf in argv_dict.items():
        argv_dict[key] = 1 + math.log(raw_tf, 10)  # now argv_dict contains the
        # log of term frequency

    # For each token in the query, find in the database:
    #   If there is none in the database: say there's none
    #   If there is: calculate the score, q_squared, d_squared of each doc_id to later calculate cosine similarity
    for key, value in argv_dict.items():
        # Get the token from MongoDB database into var
        var = writer.find_one({"word": key})
        if var is None:
            print(key, " is not in our database")
        else:
            # Get the doc_id, tf_idf properties of the token in the database
            meta_data = var.get("meta_data")
            doc_freq = len(meta_data.keys())  # get the length of posting list of that token
            for doc_id in meta_data.keys():
                doc_tf_idf = meta_data[doc_id]
                # already logged query_term_frequencies * log(total_doc / doc_frequencies)
                term_tf_idf = value * math.log(num_doc / doc_freq, 10)

                score[doc_id] += doc_tf_idf * term_tf_idf
                q_squared[doc_id] += math.pow(term_tf_idf, 2)
                d_squared[doc_id] += math.pow(doc_tf_idf, 2)

    # Calculate consine similarity for the url, then store the top_20 relevant url
    # other_url contains the consine similarity for other sites that do not have as
    # good as a score as top_20
    top_20 = defaultdict(float)
    other_url = defaultdict(float)
    for doc_id in score.keys():
        score[doc_id] = score[doc_id] / (math.sqrt(q_squared[doc_id]) * math.sqrt(d_squared[doc_id]))
        if score[doc_id] == 1:
            top_20.update({doc_id: score[doc_id]})
            if len(top_20) >= 20:
                break
        else:
            other_url.update({doc_id: score[doc_id]})

    # In case of top 20 doesn't have enough members, then take top members from
    # other_url to max out top_20 list
    num_missing = 20 - len(top_20)
    other_url = dict(sorted(other_url.items(), key=lambda x: -x[1]))
    i = 0
    if num_missing > 0:
        for key, value in other_url.items():
            top_20.update({key: value})
            i += 1
            if i >= num_missing:
                break

    # Print top_20
    for doc_id, cosine_score in top_20.items():
        print(doc_id, " ", cosine_score, " ", url_corpus[doc_id])

    # for doc_id, cosine_score in score.items():
    #    print(doc_id, " ", cosine_score, " ", url_corpus[doc_id])
