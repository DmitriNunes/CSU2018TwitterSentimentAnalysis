import pandas as pd
import numpy as np
import csv
import nltk
import string
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
import random

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

import pickle
import re
from collections import Counter
from string import punctuation
import tweepy
from tweepy import OAuthHandler
import json
from wordcloud import WordCloud

import plotly
import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()

from IPython.display import IFrame
import folium
from folium import plugins
from folium.plugins import MarkerCluster, FastMarkerCluster, HeatMapWithTime

pd.set_option('display.max_colwidth', -1)
plt.style.use('seaborn-white')

from datetime import datetime

# Load tweets from CSV, save as pickle and return pickle file name
def loadTweets(file):

	pd.set_option('display.max_colwidth', -1)
	plt.style.use('seaborn-white')
	

	col_names = ['id','text']
	data = pd.read_csv(file, error_bad_lines=False, encoding='latin-1', names=col_names)

	
	current_date = datetime.now().strftime("%Y%m%d")
	filename = 'df_tweets_' + current_date + '.p'
	data.to_pickle(filename) # save dframe to pickle
	return filename




def runML(pfile):
	
	model_NB = joblib.load("twitter_sentiment.pkl" )
	df_twtr = pd.read_pickle(pfile)  # load from pickle
	tweet_preds = model_NB.predict(df_twtr['text'])
	df_tweet_preds = df_twtr.copy()
	df_tweet_preds['predictions'] = tweet_preds
	df_tweet_preds.to_csv('final_results.csv', index=False)
	pos = df_tweet_preds.predictions.value_counts(4)
	neg = df_tweet_preds.predictions.value_counts(0)

	print('Model predictions: Positives - {}, Negatives - {}'.format(pos,neg))
	index = random.sample(range(tweet_preds.shape[0]), 20)
	for text, sentiment in zip(df_tweet_preds.text[index],
								df_tweet_preds.predictions[index]):
		print(sentiment, '--', text, '\n')
	
	df_tweet_preds.to_pickle('predicts_df.p')
	df = pd.read_pickle('predicts_df.p')
	
	def lats(x):
		return x[1]

	def longs(x):
		return x[0]

	# --------------------------------------------------------#
	# append longs and lats to dframe
	df['latitude'] = df['geo_code'].apply(lats)
	df['longitude'] = df['geo_code'].apply(longs)
	df.columns


	# for US tweets extract state abreviations for a new STATE column
	# helper function to extract state origin of every tweet
	def get_state(x):
    
		states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
				"HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
				"MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
				"NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
				"SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

		states_dict = {
				'AK': 'Alaska','AL': 'Alabama','AR': 'Arkansas','AS': 'American Samoa',
				'AZ': 'Arizona','CA': 'California','CO': 'Colorado','CT': 'Connecticut',
				'DC': 'District of Columbia','DE': 'Delaware','FL': 'Florida','GA': 'Georgia',
				'GU': 'Guam','HI': 'Hawaii','IA': 'Iowa','ID': 'Idaho','IL': 'Illinois',
				'IN': 'Indiana','KS': 'Kansas','KY': 'Kentucky','LA': 'Louisiana',
				'MA': 'Massachusetts','MD': 'Maryland','ME': 'Maine','MI': 'Michigan',
				'MN': 'Minnesota','MO': 'Missouri','MP': 'Northern Mariana Islands',
				'MS': 'Mississippi','MT': 'Montana','NA': 'National','NC': 'North Carolina',
				'ND': 'North Dakota','NE': 'Nebraska','NH': 'New Hampshire','NJ': 'New Jersey',
				'NM': 'New Mexico','NV': 'Nevada','NY': 'New York','OH': 'Ohio','OK': 'Oklahoma',
				'OR': 'Oregon','PA': 'Pennsylvania','PR': 'Puerto Rico','RI': 'Rhode Island',
				'SC': 'South Carolina','SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas',
				'UT': 'Utah','VA': 'Virginia','VI': 'Virgin Islands','VT': 'Vermont',
				'WA': 'Washington','WI': 'Wisconsin','WV': 'West Virginia','WY': 'Wyoming'
		}


		abv = x.split(',')[-1].lstrip().upper()
		state_name = x.split(',')[0].lstrip()
		if abv in states:
			state = abv
		else:
			if state_name in states_dict.values():
				state = list(states_dict.keys())[list(states_dict.values()).index(state_name)]
			else:
				state = 'Non_USA'    
		return state

	# ____________________________________________________________________________

	# create abreviated states column
	df = df.copy()
	df['states'] = df['full_name'].apply(get_state)
	list(df['states'].head())
	df_states = df[df.country=='United States']
	df_states = df_states[df_states.states!='Non_USA']
	# use the folium library to create all tweet origins in the dataset on map of US

	geoplots = []
	for index, row in df_states[['latitude','longitude','predictions']].iterrows():
		geoplots.append([row['latitude'],row['longitude'],row['predictions']])

	mus = folium.Map(location=[39, -99], zoom_start=4)
	plugins.Fullscreen(
		position='topright',
		title='Expand me',
		title_cancel='Exit me',
		force_separate_button=True).add_to(mus)

	mus.choropleth(
		geo_data='us_states.geojson',
		fill_color='red', 
		fill_opacity=0.1, 
		line_opacity=0.2,
		name='US States')
    
	mus.add_child(plugins.HeatMap(geoplots,
								name='Twitter HeatMap',
								radius=10,
								max_zoom=1,
								blur=10, 
								max_val=3.0))
	folium.TileLayer('cartodbpositron').add_to(mus)
	folium.TileLayer('cartodbdark_matter').add_to(mus)
	folium.TileLayer('Mapbox Control Room').add_to(mus)
	folium.LayerControl().add_to(mus)
	mus.save("twitter_us_map.html") 
	IFrame('twitter_us_map.html', width=960, height=520)

# some code to save and display image to png  for all browsers support
	img = plt.imread('twitter_sentiment_crop.png')
	plt.figure(figsize=(18,9))
	plt.imshow(img)