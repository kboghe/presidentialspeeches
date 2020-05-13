##################
#import libraries#
##################

#nlp framework#
import nltk

#sentiment libraries#
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pattern.en import sentiment
from flair.models import TextClassifier
from flair.data import Sentence
#pandas#
import pandas as pd
#tools#
import re
from langdetect import detect
from tqdm import tqdm
#visualization#
import matplotlib as mpl
import matplotlib.pyplot as plt
from graphics import comparative_graph
import numpy as np
from matplotlib.lines import Line2D
#stats#
from scipy.stats import mannwhitneyu
import statistics

#####################
# sentiment function#
#####################

def sentiment_df(df,idcolumn,textcolumn):
  rank_sentence,sentence_list,id_list, sentiment_vader_list, sentiment_pattern_list,sub_pattern_list, \
  neutral_list, neg_list, pos_list, language = ([] for i in range(8))
  for index, row in tqdm(df.iterrows()):
    parsed = nltk.tokenize.sent_tokenize(row[textcolumn])
    regex_delete = re.compile(r'\(.*\)$|\.')
    parsed = [i for i in parsed if not regex_delete.match(i)]
    rank = 0
    print("Performing sentiment analysis...\n")
    for element in parsed:
      sent_vader_sentence = list(SentimentIntensityAnalyzer().polarity_scores(element).values())
      sent_pattern_sentence = sentiment(element)
      sentiment_pattern_list.append(sent_pattern_sentence[0])
      sub_pattern_list.append(sent_pattern_sentence[1])
      sentiment_vader_list.append(sent_vader_sentence[3])
      neg_list.append(sent_vader_sentence[0])
      neutral_list.append(sent_vader_sentence[1])
      pos_list.append(sent_vader_sentence[2])
      id_list.append(row[idcolumn])
      sentence_list.append(element)
      rank_sentence.append(rank)
      rank = rank + 1
      try:
        language.append(detect(element))
      except:
        language.append("no features in text")
    output = pd.DataFrame({'id':id_list,'rank_sentence':rank_sentence,'sentence':sentence_list,'sentiment_vader': sentiment_vader_list,
                           'neutral_vader':neutral_list, 'negative_vader':neg_list,'positive_vader':pos_list,
                           'sentiment pattern':sentiment_pattern_list,'subjectivity pattern':sub_pattern_list, 'language':language}
                          ,columns=['id','rank_sentence','sentence','sentiment_vader', 'neutral_vader',
                                    'negative_vader','positive_vader','sentiment_pattern','subjectivity_pattern','language'])
  print("Done!")
  return(output)

presidential_speeches = pd.read_csv("2presidential_speeches_with_metadata.csv", sep=";",encoding="utf-8",quotechar="'")


inauguration_speeches = presidential_speeches[presidential_speeches['title'].str.contains("Inaugural")].reset_index()

sentiment_inaug = sentiment_df(inauguration_speeches,'title','speech')
sentiment_inaug = sentiment_inaug.merge(presidential_speeches.drop(columns="speech"),left_on="id", right_on="title",how="left")
sentiment_inaug = sentiment_inaug.set_index('id')
sentiment_inaug['cumcount'] = sentiment_inaug.groupby("id",sort=False).cumcount()+1
totalcount = sentiment_inaug.groupby("id")['cumcount'].count().reset_index().rename(columns={'cumcount':'count'})
sentiment_inaug = sentiment_inaug.merge(totalcount,on="id",how="left")
sentiment_inaug['percentage'] = sentiment_inaug['cumcount']/sentiment_inaug['count']
sentiment_inaug = sentiment_inaug.drop(columns=['count'])

#from Kennedy onwards#
sentiment_inaug_kennedy = sentiment_inaug[sentiment_inaug['from'] > 1959]
sentiment_inaug_kennedy = sentiment_inaug_kennedy.sort_values(by=['date','rank_sentence'])

#################
###create plot###
#################

comparative_graph(sentiment_inaug_kennedy)

#mean sentiment of speeches#

plt.style.use('fivethirtyeight')
mean_sent_speech = sentiment_inaug_kennedy.groupby('id').agg({'sentiment_vader': ['mean'],'Party': ['first'],'date': ['min']})
mean_sent_speech.columns = ['mean sentiment','party','year']
mean_sent_speech['year'] = mean_sent_speech['year'].str[:4].astype(int)
mean_sent_speech['mean sentiment'] = mean_sent_speech['mean sentiment'].apply(lambda x: float(x))
mean_sent_speech = mean_sent_speech.sort_values(by=['year'])

plt.plot(mean_sent_speech['year'],mean_sent_speech['mean sentiment'],linewidth=1.5)
plt.suptitle('Sentiment of inaugural speeches over time since John F. Kennedy',fontsize=10, fontweight='bold')
plt.text(0.5, 0.01, 'Year', ha='center', va='center',fontsize=7)
plt.text(0.01, 0.5, 'Sentiment (< 0: negative, > 0: positive)', ha='left', va='center', rotation = 'vertical',fontsize=7)

plt.savefig('over_time.png')


republicansent = sentiment_inaug_kennedy.sentiment_vader[sentiment_inaug_kennedy['Party'] == "Republican"]
democratsent = sentiment_inaug_kennedy.sentiment_vader[sentiment_inaug_kennedy['Party'] == "Democratic"]

u_statistic, pVal = mannwhitneyu(republicansent, democratsent)
print("The median sentiment value of the two groups are (sentence-level):")
print("for Republican sentences: "+ str((statistics.median(republicansent))))
print("for Democratic sentences: "+ str((statistics.median(democratsent)))+"\n")
print ('The P value is:')
print (pVal)