# CSU2018TwitterSentimentAnalysis
A project done for the Launchpad partnership between MailChimp and Clayton State University.
MailchimpBDTraining.py will train a Naive-Bayes Classifier model using the Sentiment140 dataset.
ml.py will take the model produced by MailchimpBDTraining.py and run it over a CSV that is provided to it.
run.py loads ml.py, sends it a CSV, and runs the classification.
twitter_sentiment.pkl is the model saved in pickle format.
results.csv is a sample CSV of tweets.
final_results.csv is the output of run.py on results.csv
