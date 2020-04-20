# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:08:29 2020

@author: Suraj Gadkar
"""
# Importing libraires
#import chromedriver_binary
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# Importing driver and getiing the url
driver = webdriver.Chrome()

# -------------------------------Repetative Code Begins-------------------------------------------------#
# Instructions for scraping each catgorical data
""" 1. Change the Respected Code in the driver.get(---change--here--)
    2. Run the importing driver code
    3. Run the Repetative code section"""
# For entertainment 
# Creating the data frame
 driver.get("https://www.youtube.com/results?search_query=science&sp=EgIQAQ%253D%253D")

# fetchign the links from the url
user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
link_enter = []
for i in user_data:
    link_enter.append(i.get_attribute('href'))

print(len(links))

df_entertainment = pd.DataFrame(columns = ['link', 'title', 'description', 'category'])
# Script for scraping details
# Scraper for Entertainment category
wait =  WebDriverWait(driver, 10)
v_category = "Entertainment"
for x in link_enter:
    driver.get(x)
    v_id = x.strip('https://www.youtube.com/watch?v=')
    v_title = wait.until(EC.visibility_of_element_located(
            (By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
    v_description = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR,"div #description"))).text
    df_entertainment.loc[len(df_entertainment)] = [v_id, v_title, v_description, v_category]
    
# For travel_vlogs 
driver.get("Respected url")
# paste travel_vlogs url
# fetchign the links from the url
user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
link_travel = []
for i in user_data:
    link_travel.append(i.get_attribute('href'))

print(len(link_travel))

# Creating the data frame
df_travel_vlogs = pd.DataFrame(columns = ['link', 'title', 'description', 'category'])
# Script for scraping details
# Scraper for Travel Vlogs category
wait =  WebDriverWait(driver, 10)
v_category = "Travel_vlogs"
for x in link_travel:
    driver.get(x)
    v_id = x.strip('https://www.youtube.com/watch?v=')
    v_title = wait.until(EC.visibility_of_element_located(
            (By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
    v_description = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR,"div #description"))).text
    df_travel_vlogs.loc[len(df_travel_vlogs)] = [v_id, v_title, v_description, v_category]
    
# For  science 
driver.get("https://www.youtube.com/results?search_query=science&sp=EgIQAQ%253D%253D")

# fetchign the links from the url
user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
link_science = []
for i in user_data:
    link_science.append(i.get_attribute('href'))

print(len(link_science))

# Creating the data frame
df_science = pd.DataFrame(columns = ['link', 'title', 'description', 'category'])
# Script for scraping details
# Scraper for Travel Vlogs category
wait =  WebDriverWait(driver, 10)
v_category = "Science"
for x in link_science:
    driver.get(x)
    v_id = x.strip('https://www.youtube.com/watch?v=')
    v_title = wait.until(EC.visibility_of_element_located(
            (By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
    v_description = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR,"div #description"))).text
    df_science.loc[len(df_science)] = [v_id, v_title, v_description, v_category]

# For  food 
# Creating the data frame
    # paste food url
driver.get("https://www.youtube.com/results?search_query=food&sp=EgIQAQ%253D%253D")

# fetchign the links from the url
user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
link_food = []
for i in user_data:
    link_food.append(i.get_attribute('href'))

print(len(link_food))

df_food  = pd.DataFrame(columns = ['link', 'title', 'description', 'category'])
# Script for scraping details
# Scraper for Travel Vlogs category
wait =  WebDriverWait(driver, 10)
v_category = "Food "
for x in link_food:
    driver.get(x)
    v_id = x.strip('https://www.youtube.com/watch?v=')
    v_title = wait.until(EC.visibility_of_element_located(
            (By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
    v_description = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR,"div #description"))).text
    df_food.loc[len(df_food )] = [v_id, v_title, v_description, v_category]
    
# For  history 
# paste history url
driver.get("https://www.youtube.com/results?search_query=history&sp=EgIQAQ%253D%253D")

# fetchign the links from the url
user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
link_history = []
for i in user_data:
    link_history.append(i.get_attribute('href'))

print(len(link_history))
# Creating the data frame
df_history = pd.DataFrame(columns = ['link', 'title', 'description', 'category'])
# Script for scraping details
# Scraper for Travel Vlogs category
wait =  WebDriverWait(driver, 10)
v_category = "History"
for x in link_history:
    driver.get(x)
    v_id = x.strip('https://www.youtube.com/watch?v=')
    v_title = wait.until(EC.visibility_of_element_located(
            (By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
    v_description = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR,"div #description"))).text
    df_history.loc[len(df_history)] = [v_id, v_title, v_description, v_category]
    
# For  manufacturing 
# Creating the data frame
     #paste manufacturing url
driver.get("https://www.youtube.com/results?search_query=manufacture&sp=EgIQAQ%253D%253D")

# fetchign the links from the url
user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')
link_manu = []
for i in user_data:
    link_manu.append(i.get_attribute('href'))

   
df_manufacturing = pd.DataFrame(columns = ['link', 'title', 'description', 'category'])
# Script for scraping details
# Scraper for Travel Vlogs category
wait =  WebDriverWait(driver, 10)
v_category = "Manufacturing"
for x in link_manu:
    driver.get(x)
    v_id = x.strip('https://www.youtube.com/watch?v=')
    v_title = wait.until(EC.visibility_of_element_located(
            (By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text
    v_description = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR,"div #description"))).text
    df_manufacturing.loc[len(df_manufacturing)] = [v_id, v_title, v_description, v_category]
    
    
# -------------------------------Repetative Code Ends-------------------------------------------------#
   
# Merging all dataframes into a dataset
frames = [df_travel_vlogs, df_science, df_manufacturing, df_history, df_food, df_entertainment]
dataframe = pd.concat(frames, axis = 0, join = 'outer', join_axes = None, ignore_index = True,
                      keys= None, levels = None, names = None, verify_integrity = False, copy = True)
    
# Copying each column in dataframe as a separate dataframe.
df_link = pd.DataFrame(columns = ["link"])
df_title = pd.DataFrame(columns = ["title"])
df_description = pd.DataFrame(columns = ["description"])
df_category = pd.DataFrame(columns = ["category"])

df_link['link'] = dataframe['link']
df_title['title'] = dataframe['title']
df_description['description'] = dataframe['description']
df_category['category'] = dataframe['category']

#-------------------------------- Cleaning the dataset using NLTK toolkit-----------------------------
#importing libraries
import re
import nltk
nltk.download('stopwotds')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# create a list where we will store the cleaned text
corpus_title = []
for i in range(0, 117):
    review = re.sub('[^a-zA-Z]', ' ', df_title['title'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_title.append(review)


corpus_description = []
for x in range(0, 117):
    review = re.sub('[^a-zA-Z]', ' ', df_description['description'][x])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_description.append(review)
    
# Convert the list into dataframes
dftitle = pd.DataFrame({'title':corpus_title})
dfdescription = pd.DataFrame({'description':corpus_description})

#LabelEncoding
from sklearn.preprocessing import LabelEncoder
dfcategory = df_category.apply(LabelEncoder().fit_transform)

#Merging all the columns to create a cleaned dataframe
df_new = pd.concat([df_link, dftitle, dfdescription, dfcategory], axis=1, join_axes = [df_link.index])

# Creating the 'Bag Of Words' Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus_title, corpus_description).toarray()
y = df_new.iloc[:, 3].values

#------------------------ Fitting the dataframe to the classification model---------------------------

# Splitting the dataframe into train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1500, criterion = 'entropy')
classifier.fit(X_train,y_train)


# Predicting using the model
y_pred = classifier.predict(X_test)
score = (classifier.score(X_test, y_test))

#-----------------------------------Analysing Report--------------------------------------------------
# printing classfication report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
"""   The Classfication Report  
         precision    recall  f1-score   support

           0       1.00      0.75      0.86         4
           1       1.00      1.00      1.00         5
           2       0.80      1.00      0.89         4
           3       1.00      1.00      1.00         3
           4       1.00      1.00      1.00         3
           5       1.00      1.00      1.00         5

    accuracy                           0.96        24
   macro avg       0.97      0.96      0.96        24
weighted avg       0.97      0.96      0.96        24"""

cm = confusion_matrix(y_pred, y_test)









 
    
    