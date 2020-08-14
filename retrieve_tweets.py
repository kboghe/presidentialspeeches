import time
import pandas as pd
import GetOldTweets3 as got


def get_tweets(query,maxtweet):
    tweetCriteria = got.manager.TweetCriteria()\
        .setUsername(query)\
        .setMaxTweets(int(maxtweet))
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    idslist, tweetslist = ([] for i in range(2))
    for tweetelement in tweets:
        idslist.append(tweetelement.id)
        tweetslist.append(tweetelement.text)
    tweets_final = pd.DataFrame({'id': idslist, 'tweet': tweetslist})
    tweets_final['tweet'] = tweets_final['tweet'].replace("http\S+|#\S+|@\S+","",regex=True).apply(lambda x: x.strip())
    tweets_final = tweets_final.drop_duplicates(subset="tweet")
    return(tweets_final)

queries = ['realDonaldTrump','AOC','SpeakerPelosi','BernieSanders','RandPaul','tedcruz']
tweets_dataset = pd.DataFrame()
for query in queries:
    print("Fetching tweets from "+query+"...")
    tweets = get_tweets(query,"2000")
    tweets_dataset = pd.concat([tweets_dataset,tweets])
    print("Done! Pausing for a minute...\n")
    time.sleep(60)