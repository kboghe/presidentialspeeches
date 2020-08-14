#nlp framework#
import nltk
#sentiment libraries#
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #VADER
from pattern.en import sentiment #Pattern
from pycorenlp import StanfordCoreNLP
#define server for StanfordCoreNLP#
#pandas#
import pandas as pd
#tools#
import re
from langdetect import detect
from tqdm import tqdm

##################################################
def sentiment_df(df,idcolumn,textcolumn,sentence_level = 0):
    nlp = StanfordCoreNLP('http://localhost:9000')
    rank_sentence,sentence_list,id_list, sentiment_stanford_list, sentiment_vader_list, sentiment_pattern_list,sub_pattern_list, \
    neutral_list, neg_list, pos_list, language = ([] for i in range(11))
    for index, row in tqdm(enumerate(df.iterrows())):
        if sentence_level == 1:
            parsed = nltk.tokenize.sent_tokenize(str(row[textcolumn]))
            regex_delete = re.compile(r'\(.*\)$|\.')
            parsed = [i for i in parsed if not regex_delete.match(i)]
        else:
            parsed = list([str(row[textcolumn])])
        print("Performing sentiment analysis...\n")
        for index,element in enumerate(parsed):
            sent_vader_sentence = list(SentimentIntensityAnalyzer().polarity_scores(element).values())
            sent_pattern_sentence = sentiment(element)
            sentiment_pattern_list.append(sent_pattern_sentence[0])
            sub_pattern_list.append(sent_pattern_sentence[1])
            sentiment_vader_list.append(sent_vader_sentence[3])
            neg_list.append(sent_vader_sentence[0])
            neutral_list.append(sent_vader_sentence[1])
            pos_list.append(sent_vader_sentence[2])
            sentiment_stanford = nlp.annotate(element, properties={'timeout': '500000','annotators': 'sentiment', 'outputFormat': 'json'})
            sentiment_stanford = sentiment_stanford['sentences'][0]['sentimentValue']
            sentiment_stanford_list.append(sentiment_stanford)
            id_list.append(row[idcolumn])
            sentence_list.append(element)
            rank_sentence.append(index)
            try:
                language.append(detect(element))
            except:
                language.append("no features in text")
        output = pd.DataFrame({'id': id_list, 'rank_sentence': rank_sentence, 'sentence': sentence_list,
                               'sentiment_stanford':sentiment_stanford_list,'sentiment_vader': sentiment_vader_list,
                               'neutral_vader': neutral_list, 'negative_vader': neg_list, 'positive_vader': pos_list,
                               'sentiment_pattern': sentiment_pattern_list, 'subjectivity_pattern': sub_pattern_list,
                               'language': language})
        output = output.merge(df.drop(columns=textcolumn), left_on="id",
                              right_on=idcolumn, how="left").set_index('id')
    print("Done!")
    return (output)


#####################################
def speech_progress(df):
    df['cumcount'] = df.groupby("id", sort=False).cumcount() + 1
    totalcount = df.groupby("id")['cumcount'].count().reset_index().rename(columns={'cumcount': 'count'})
    df = df.merge(totalcount, on="id", how="left")
    df['percentage'] = df['cumcount'] / df['count']
    df = df.drop(columns=['count'])
    return df