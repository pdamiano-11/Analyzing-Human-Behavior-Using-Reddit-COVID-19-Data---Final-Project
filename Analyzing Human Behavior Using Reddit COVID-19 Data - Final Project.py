# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:27:15 2020

@author: ptrda
"""

import praw
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import sentiwordnet as swn
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Opening the CONFIG file and getting all sensitive info.

file = open("CONFIG.txt", "r")
config = file.readlines()
file.close()

client_ID = config[0].replace("\n", "")
secret = config[1].replace("\n", "")
user_name = config[2].replace("\n", "")
passwd = config[3].replace("\n", "")

# Calling to the Reddit API and inputting the sensitive info.

reddit = praw.Reddit(client_id=client_ID, \
                     client_secret=secret, \
                     user_agent='Coronavirus Scraping tool', \
                     username=user_name, \
                     password=passwd)

# Setting up the subreddit and getting the number of posts.

covid_subreddit = reddit.subreddit('COVID19_support')

top_posts_covid = covid_subreddit.top("month", limit = 30)

# Setting up a dictionary to get several components of each post and convert into a data frame.

posts_dict = { "title":[], \
                "score":[], \
                "id":[], \
                "comms_num": [], \
                "created": [], \
                "body":[]}

for submission in top_posts_covid:
    posts_dict["title"].append(submission.title)
    posts_dict["score"].append(submission.score)
    posts_dict["id"].append(submission.id)
    posts_dict["comms_num"].append(submission.num_comments)
    posts_dict["created"].append(submission.created)
    posts_dict["body"].append(submission.selftext)

posts_data = pd.DataFrame(posts_dict)

# Getting the top 10 comments per post.

posts_data["Top 10 Comments"] = np.nan
for i in range(0, len(posts_data.id)):
    app_com = []
    s = reddit.submission(posts_data.id[i])
    s.comment_sort = "top"
    comments = s.comments
    if len(comments) >= 10:
        for c in range(9):
            app_com.append(comments[c].body)
    posts_data["Top 10 Comments"][i] = app_com

# Removing variables containing sensitive info.

del user_name
del passwd
del secret
del client_ID

# Quick info on the data frame.

print(posts_data.head())
print("The number of lines in the dataset is: ", len(posts_data))

# Determining Sentiment score per post based on title.

posts_data["Post Sentiment Score"] = np.nan
for i in range(0, len(posts_data["title"])):
    title_words = posts_data["title"][i].replace(':', '').replace('.', '').replace(',', '')\
               .replace('-', ' ').replace('(', '').replace(')', '').replace("'", "")\
               .replace('“', '').replace('”', '').replace('!', '').lower()        
    
    token_title = nltk.word_tokenize(title_words)
    tag_title = nltk.pos_tag(token_title)
    score_pos = 0
    score_neg = 0
    for word in tag_title:
        if word[1] == 'JJ':
            try:
                swn.senti_synset(word[0] + '.a.01')
            except:
                continue
            score_pos += swn.senti_synset(word[0] + '.a.01').pos_score()
            score_neg += swn.senti_synset(word[0] + '.a.01').neg_score()
    if score_neg < score_pos:
        posts_data["Post Sentiment Score"][i] = score_pos
    else:
        if score_neg == 0:
            posts_data["Post Sentiment Score"][i] = score_neg
        else:
            posts_data["Post Sentiment Score"][i] = score_neg*-1

#Sorting based on sentiment score and splitting into positive, neutral, and negative scores.

posts_data = posts_data.sort_values(by = 'Post Sentiment Score', ascending = False)
posts_data = posts_data.reset_index(drop = True)
    
pos_posts = pd.DataFrame(posts_data[0:10])
neutral_posts = pd.DataFrame(posts_data[10:20])
neg_posts = pd.DataFrame(posts_data[20:30])

pos_posts = pos_posts.reset_index(drop = True)
neutral_posts = neutral_posts.reset_index(drop = True)
neg_posts = neg_posts.reset_index(drop = True)

#Calculating mean score per sub-dataset.

print("\nThe mean score for the positive posts is ", np.mean(pos_posts["score"]))
print("\nThe mean score for the neutral posts is ", np.mean(neutral_posts["score"]))
print("\nThe mean score for the negative posts is ", np.mean(neg_posts["score"]))

# Determing comment sentimental scores.

poscomm_score_pos = 0
poscomm_score_neg = 0
poscomm = ""
for i in range(0, len(pos_posts["Top 10 Comments"])):
    for r in range(0, len(pos_posts["Top 10 Comments"][i])):
        poscomm += pos_posts["Top 10 Comments"][i][r] + " "
        
text_poscomm = poscomm.replace(':', '').replace('.', '').replace(',', '')\
       .replace('-', ' ').replace('(', '').replace(')', '').replace("[", "").replace("]", "")\
       .replace('“', '').replace('”', '').replace('!', '').lower()        
    
token = nltk.word_tokenize(text_poscomm)
tag = nltk.pos_tag(token)
for word in tag:
    if word[1] == 'JJ':
        try:
            swn.senti_synset(word[0] + '.a.01')
        except:
            continue
        poscomm_score_pos += swn.senti_synset(word[0] + '.a.01').pos_score()
        poscomm_score_neg += swn.senti_synset(word[0] + '.a.01').neg_score()

print("\nFor positive posts, the positive comment sentiment score is ", poscomm_score_pos)
print("\nFor positive posts, the negative comment sentiment score is ", poscomm_score_neg)
    
    
neutcomm_score_pos = 0
neutcomm_score_neg = 0
neutcomm = ""
for i in range(0, len(neutral_posts["Top 10 Comments"])):
    for r in range(0, len(neutral_posts["Top 10 Comments"][i])):
        neutcomm += neutral_posts["Top 10 Comments"][i][r] + " "
        
text_neutcomm = neutcomm.replace(':', '').replace('.', '').replace(',', '')\
       .replace('-', ' ').replace('(', '').replace(')', '').replace("[", "").replace("]", "")\
       .replace('“', '').replace('”', '').replace('!', '').lower()        
    
token = nltk.word_tokenize(text_neutcomm)
tag = nltk.pos_tag(token)
for word in tag:
    if word[1] == 'JJ':
        try:
            swn.senti_synset(word[0] + '.a.01')
        except:
            continue
        neutcomm_score_pos += swn.senti_synset(word[0] + '.a.01').pos_score()
        neutcomm_score_neg += swn.senti_synset(word[0] + '.a.01').neg_score()

print("\nFor neutral posts, the positive comment sentiment score is ", neutcomm_score_pos)
print("\nFor neutral posts, the negative comment sentiment score is ", neutcomm_score_neg)


negcomm_score_pos = 0
negcomm_score_neg = 0
negcomm = ""
for i in range(0, len(neg_posts["Top 10 Comments"])):
    for r in range(0, len(neg_posts["Top 10 Comments"][i])):
        negcomm += neg_posts["Top 10 Comments"][i][r] + " "
        
text_negcomm = negcomm.replace(':', '').replace('.', '').replace(',', '')\
       .replace('-', ' ').replace('(', '').replace(')', '').replace("[", "").replace("]", "")\
       .replace('“', '').replace('”', '').replace('!', '').lower()        
    
token = nltk.word_tokenize(text_negcomm)
tag = nltk.pos_tag(token)
for word in tag:
    if word[1] == 'JJ':
        try:
            swn.senti_synset(word[0] + '.a.01')
        except:
            continue
        negcomm_score_pos += swn.senti_synset(word[0] + '.a.01').pos_score()
        negcomm_score_neg += swn.senti_synset(word[0] + '.a.01').neg_score()

print("\nFor negative posts, the positive comment sentiment score is ", negcomm_score_pos)
print("\nFor negative posts, the negative comment sentiment score is ", negcomm_score_neg)   


# Bringing in stopwords.

stopwords = []
file = open("stopwords_en.txt", "r")
for line in file:
    stopwords.append(line.replace('\n', ''))

# Preparing title text and comment text for word cloud generation.

postitle_text = ''
for i in range(0, len(pos_posts["title"])):
    postitle_text += pos_posts["title"][i] + " "
    
postitle_list = postitle_text.replace(':', '').replace('.', '').replace(',', '')\
           .replace('-', ' ').replace('(', '').replace(')', '')\
           .replace('“', '').replace('”', '').replace('!', '').replace('#', '').lower().strip().split()
rm_stop_postitle = []
for word in postitle_list: 
    try:
        int(word)
    except:
        if word not in stopwords:
            if word.isalpha():
                rm_stop_postitle.append(word)

postitle_wc = ' '.join(rm_stop_postitle)
    
poscomm_list = text_poscomm.strip().split()
rm_stop_poscomm = []
for word in poscomm_list: 
    try:
        int(word)
    except:
        if word not in stopwords:
            if word.isalpha():
                rm_stop_poscomm.append(word)

poscomm_wc = ' '.join(rm_stop_poscomm)


neuttitle_text = ''
for i in range(0, len(neutral_posts["title"])):
    neuttitle_text += neutral_posts["title"][i] + " "
    
neuttitle_list = neuttitle_text.replace(':', '').replace('.', '').replace(',', '')\
           .replace('-', ' ').replace('(', '').replace(')', '')\
           .replace('“', '').replace('”', '').replace('!', '').replace('#', '').lower().strip().split()
rm_stop_neuttitle = []
for word in neuttitle_list: 
    try:
        int(word)
    except:
        if word not in stopwords:
            if word.isalpha():
                rm_stop_neuttitle.append(word)

neuttitle_wc = ' '.join(rm_stop_neuttitle)
    
neutcomm_list = text_neutcomm.strip().split()
rm_stop_neutcomm = []
for word in neutcomm_list: 
    try:
        int(word)
    except:
        if word not in stopwords:
            if word.isalpha():
                rm_stop_neutcomm.append(word)

neutcomm_wc = ' '.join(rm_stop_neutcomm)


negtitle_text = ''
for i in range(0, len(neg_posts["title"])):
    negtitle_text += neg_posts["title"][i] + " "
    
negtitle_list = negtitle_text.replace(':', '').replace('.', '').replace(',', '')\
           .replace('-', ' ').replace('(', '').replace(')', '')\
           .replace('“', '').replace('”', '').replace('!', '').replace('#', '').lower().strip().split()
rm_stop_negtitle = []
for word in negtitle_list: 
    try:
        int(word)
    except:
        if word not in stopwords:
            if word.isalpha():
                rm_stop_negtitle.append(word)

negtitle_wc = ' '.join(rm_stop_negtitle)
    
negcomm_list = text_negcomm.strip().split()
rm_stop_negcomm = []
for word in negcomm_list: 
    try:
        int(word)
    except:
        if word not in stopwords:
            if word.isalpha():
                rm_stop_negcomm.append(word)

negcomm_wc = ' '.join(rm_stop_negcomm)

# Generating all SIX word clouds.

wc_title = WordCloud(height = 2000, width = 3000, background_color='white', max_words = 100, collocations = False)
wc_comm = WordCloud(height = 2000, width = 3000, background_color='white', max_words = 500, collocations = False)

wc_title.generate(postitle_wc)
wc_title.to_file("Positive Titles.png")
plt.imshow(wc_title)
plt.axis('off')
plt.show()   

wc_comm.generate(poscomm_wc)
wc_comm.to_file("Comments for Positive Titles.png")
plt.imshow(wc_comm)
plt.axis('off')
plt.show()     


wc_title.generate(neuttitle_wc)
wc_title.to_file("Neutral Titles.png")
plt.imshow(wc_title)
plt.axis('off')
plt.show()   

wc_comm.generate(neutcomm_wc)
wc_comm.to_file("Comments for Neutral Titles.png")
plt.imshow(wc_comm)
plt.axis('off')
plt.show()     


wc_title.generate(negtitle_wc)
wc_title.to_file("Negative Titles.png")
plt.imshow(wc_title)
plt.axis('off')
plt.show()   

wc_comm.generate(negcomm_wc)
wc_comm.to_file("Comments for Negative Titles.png")
plt.imshow(wc_comm)
plt.axis('off')
plt.show()         
    
    
    
    
    
    