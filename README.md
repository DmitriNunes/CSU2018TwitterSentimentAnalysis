# CSU2018TwitterSentimentAnalysis
A project done for the Launchpad partnership between MailChimp and Clayton State University.

*twyconnect.py connects to Twitterand downloads tweets containing the keyword provided (NOTE: We removed our consumer key/secret from the code. You must set up your own Dev environment on Twitter to get one yourself, or be provided one).

*MailchimpBDTraining.py will train a Naive-Bayes Classifier model using the Sentiment140 dataset.

*ml.py will take the model produced by MailchimpBDTraining.py and run it over a CSV that is provided to it.

*run.py loads ml.py, sends it a CSV, and runs the classification.

*twitter_sentiment.pkl is the model saved in pickle format.

*results.csv is a sample CSV of tweets.

*final_results.csv is the output of run.py on results.csv

*DB_interface.py generates the interface and graph representing the final result of the sentiment analysis.

*MailChimp_Big_Data_Final_2.pptx is the presentation given at MailChimp headquarters on November 30, 2018.



By Robert Pribbenow, Floyd Askew, Dmitri Dias Fernandes, and Nguyen Kim.
