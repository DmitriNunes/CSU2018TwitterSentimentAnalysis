import tweepy
import json

def twitterConnect(queryWord='MailChimp', max_tweets = 1000):

    consumer_key = '''Your consumer key goes here'''
    consumer_secret = '''Your consumer secret goes here'''

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    api = tweepy.API(auth)

    file = open("results.csv","wt+",encoding="utf-8")

    query = queryWord
    lang = 'en'
    mode = 'extended'
    for status in tweepy.Cursor(api.search, q=query, lang=lang, tweet_mode=mode).items(max_tweets):
        temp = status._json
        tweetID = temp['id_str']

        if 'retweeted_status' in temp:
            text = 'RT @' + temp['retweeted_status']['user']['screen_name'] + " " + temp['retweeted_status']['full_text']
        else:
            text = temp['full_text']

        text = text.replace("\n","")
        text = text.replace("\"","\"\"")
        file.write(tweetID+",\""+text+"\"\n")