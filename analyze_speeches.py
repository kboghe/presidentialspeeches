##################
#import libraries#
##################

#subprocess#
import subprocess
import time
#data wrangling#
import pandas as pd
import numpy as np
from datawrangling import sentiment_df, speech_progress
import re
#visualization#
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from graphics import comparative_graph,comparative_figures,corr_info
from matplotlib.lines import Line2D
import matplotlib.image as image
import seaborn
#stats#
from scipy.stats import mannwhitneyu
import statistics

sentiment_output_state_union_inaug = pd.read_csv("sentiment_inauguration_state_union.csv", sep=";",encoding="utf-8")
sentiment_output_tweets = pd.read_csv("sentiment_output_tweets.csv", sep=";",encoding="utf-8")

###################
#####load data#####
###################
tweets = pd.read_csv("tweets_final_dataset_pol.csv",sep=";",encoding="utf-8")
presidential_speeches = pd.read_csv("2presidential_speeches_with_metadata.csv", sep=";",encoding="utf-8",quotechar="'")

###################################
#### start stanford nlpcore server#
###################################
#downloaded from https://nlp.stanford.edu/software/corenlp-backup-download.html#

subprocess.Popen(['java','-mx4g','-cp','*','edu.stanford.nlp.pipeline.StanfordCoreNLPServer'],
                 cwd= "C:\stanford-corenlp-full-2018-02-27", shell=True, stdout= subprocess.DEVNULL, stderr=subprocess.STDOUT)

#subset by inaugural speeches#
inauguration_speeches = presidential_speeches[presidential_speeches['title'].str.contains("Inaugural")].reset_index()
inauguration_state_union_inaug = presidential_speeches[presidential_speeches['title'].str.contains("Inaugural|State of the Union")].reset_index()

#create sentiment dataframe
sentiment_output = sentiment_df(inauguration_speeches,'title', 'speech',sentence_level=1)
sentiment_output_state_union_inaug = sentiment_df(inauguration_state_union_inaug,'title','speech',sentence_level=1)
sentiment_inaug = sentiment_output_state_union_inaug[sentiment_output_state_union_inaug['title'].str.contains("Inaugural")].reset_index()

sentiment_output_tweets = sentiment_df(tweets,'id','tweet',sentence_level=0)

#define speech progress by sentence (beginning 0% - end 100%)
sentiment_output_state_union_inaug = speech_progress(sentiment_output_state_union_inaug)
sentiment_inaug = speech_progress(sentiment_inaug)

#from Kennedy onwards#
sentiment_inaug_kennedy = sentiment_inaug[sentiment_inaug['from'] > 1959]
sentiment_inaug_kennedy = sentiment_inaug_kennedy.sort_values(by=['date','rank_sentence'])

#check distributions of sentiment#
plt.style.use('fivethirtyeight')
sentiment_metric_list = [item for item in ['sentiment_pattern','sentiment_vader','sentiment_stanford'] for i in range(2)]
party_list = ['Democratic','Republican'] * 3
color_list = ['royalblue','lightcoral'] * 3
bins = [item for item in [20,20] for i in range(2)]
y_axis = [item for item in [50,20,60] for i in range(2)]
row = [0,0,1,1,2,2]
col = [0,1,0,1,0,1]

fig, (list_ax) = plt.subplots(3,2, figsize=(15, 16),sharey=False)  # define rows and columns
fig.suptitle("Sentiment distribution on sentence-level", fontsize=25, fontweight='bold', x = 0.52)  # define general title of plot

for i in range(6):
    data= round(sentiment_output_state_union_inaug[sentiment_output_state_union_inaug['Party'] == party_list[i]][sentiment_metric_list[i]],4)
    if row[i] < 2:
        list_ax[row[i],col[i]].hist(data, weights=np.ones(len(data)) / len(data) * 100, color= color_list[i],edgecolor = 'black',bins=bins[i])
        plt.subplots_adjust(hspace=0.60)
        for tick in list_ax[row[i], col[i]].xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')
        list_ax[row[i],col[i]].xaxis.set_major_locator(mtick.MultipleLocator(0.50))
    if row[i] == 2:
        data = pd.value_counts(data).to_frame().reset_index().sort_values(by="index")
        list_ax[row[i], col[i]].bar(data['index'], data['sentiment_stanford']/sum(data['sentiment_stanford'])*100, color=color_list[i], edgecolor='black')
        list_ax[row[i], col[i]].set_xticklabels(['verr','very neg', 'neg', 'neu', 'pos', 'very pos'])
        list_ax[row[i], col[i]].yaxis.set_major_locator(mtick.MultipleLocator(20))

    list_ax[row[i], col[i]].set_ylim([0, y_axis[i]])
    list_ax[row[i], col[i]].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    if col[i] == 1:
        list_ax[row[i],col[i]].set_yticklabels('')

plt.figtext(0.52, 0.91, "pattern", ha="center", va="top",fontsize=20,fontweight="bold", color="firebrick")
plt.figtext(0.52, 0.61, "vader", ha="center", va="top",fontsize=20,fontweight="bold", color="firebrick")
plt.figtext(0.52, 0.31, "stanford", ha="center", va="top",fontsize=20,fontweight="bold", color="firebrick")

colors = ['royalblue', 'lightcoral']
linestyles = ['solid', 'solid']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle=l) for (c, l) in zip(colors, linestyles)]
labels = ['Democrat', 'Republican']
fig.legend(lines, labels, loc="center", bbox_to_anchor=(0.52, 0.94), ncol=3, frameon=False, fontsize=16)

plt.figtext(0.04, 0.01, "Sample: 21.943 sentences from 116 presidential speeches,including all inauguration speeches and State of the Union addresses", ha="left", va="bottom",fontsize=13, fontweight="bold", color="dimgray")

plt.savefig("distribution sentiment check.png")
plt.clf()

############################
plt.style.use('fivethirtyeight')
sentiment_metric_list = ['sentiment_pattern','sentiment_vader','sentiment_stanford']
fig, (list_ax) = plt.subplots(1,3, figsize=(30,10),sharey=False)  # define rows and columns
fig.subplots_adjust(hspace=.50,top = 0.70)
fig.suptitle("Sentiment score distribution of three different models", fontsize=27, fontweight='bold', x = 0.50)  # define general title of plot
plt.figtext(0.50, 0.915, "comparing different text types", ha="center", va="top",fontsize=21,fontweight="bold", color="dimgray")
plt.figtext(0.21, 0.75, "pattern", ha="center", va="top",fontsize=20, fontweight="bold", color="firebrick")
plt.figtext(0.515, 0.75, "vader", ha="center", va="top",fontsize=20, fontweight="bold", color="firebrick")
plt.figtext(0.83, 0.75, "stanford", ha="center", va="top",fontsize=20, fontweight="bold", color="firebrick")


for index,library in enumerate(sentiment_metric_list):
    data1 = sentiment_output_state_union_inaug[library]
    data1label = ["Presidential speeches"] * len(data1)
    data2 = sentiment_output_tweets[library]
    data2label = ["Tweets"] * len(data2)

    datamerged = data1.append(data2)
    labelmerged = data1label + data2label
    data = pd.DataFrame({'sentiment_score':datamerged,'texttype':labelmerged})

    for texttype in ['Presidential speeches','Tweets']:
        if index < 2:
            seaborn.distplot(data['sentiment_score'][data['texttype'] == texttype], hist = False, kde = True,
                             kde_kws = {'shade': True, 'linewidth': 3},label=texttype,ax=list_ax[index])
        if index == 2:
            groupsizes = pd.DataFrame(data.groupby(['texttype']).size()).reset_index()
            groupsizes.columns = ['texttype','groupsize']
            datafinal = pd.DataFrame(data.groupby(['sentiment_score','texttype']).size()).reset_index()
            datafinal.columns = ['sentiment_score','texttype','n']
            datafinal = pd.merge(datafinal,groupsizes,how="left")
            datafinal['perc'] = datafinal['n']/datafinal['groupsize']
            seaborn.barplot(x="sentiment_score", y="perc", hue="texttype", data=datafinal,ax=list_ax[index])
            list_ax[index].set_xticklabels(['very neg','neg','neutral','pos','very pos'])

    list_ax[index].legend_.remove()
    list_ax[index].tick_params(labelsize=16)
    if index < 2:
        list_ax[index].set(xlabel='',ylabel='density')
    if index == 2:
        list_ax[index].set(xlabel='',ylabel='proportion')

handles, labels = list_ax[2].get_legend_handles_labels()
fig.legend(handles[0:2], labels[0:2], loc='upper center',ncol=2,fontsize=19,frameon=False,bbox_to_anchor=(0.50, 0.865))
fig.subplots_adjust(bottom=0.16)
plt.figtext(0.06, 0.04, "Sample: 11.467 tweets from Donald Trump, Ted Cruz, Rand Paul, Alexandria Ocasio-Cortez, Bernie Sanders & Nancy Pelosi. Retweets excluded, duplicates removed.", ha="left", va="bottom",fontsize=14, fontweight="bold", color="dimgray")
plt.figtext(0.06, 0.01, "Sample: 21.943 sentences from 116 presidential speeches, including all inauguration speeches and State of the Union addresses", ha="left", va="bottom",fontsize=14, fontweight="bold", color="dimgray")
plt.savefig("distribution sentiment comparison.png")
plt.clf()

#check agreement between libraries#

##prepare data##
speeches_figures = comparative_figures(sentiment_output_state_union_inaug)
tweets_figures = comparative_figures(sentiment_output_tweets)

speeches_corrinfo = corr_info(sentiment_output_state_union_inaug)
tweets_corrinfo = corr_info(sentiment_output_tweets)
dueto_dis_speeches = round(speeches_corrinfo[1].iloc[1,0],2)
dueto_dis_tweets = round(tweets_corrinfo[1].iloc[1,0],2)
cor_speeches = round(speeches_corrinfo[0].iloc[1,0],2)
cor_tweets = round(tweets_corrinfo[0].iloc[1,0],2)

####create figure####
plt.style.use('fivethirtyeight')
fig, (list_ax) = plt.subplots(3,4, figsize=(30,27), sharey=False)  # define rows and columns
fig.subplots_adjust(hspace=.40,top = 0.83)
fig.suptitle("(Dis)agreement between sentiment libraries", fontsize=38, fontweight='bold')  # define general title of plot
plt.figtext(0.50, 0.95, "comparing different text types", ha="center", va="top",fontsize=28,fontweight="bold", color="dimgray")

#column 0 and 2: heatmaps#
columns = [0,2]
dataframes = [tweets_figures,speeches_figures]
for index, element in enumerate(dataframes):
    heat = seaborn.heatmap(element[2]['crosstab_vader_pattern'],annot=False, cmap="plasma",ax= list_ax[0,columns[index]],cbar=True, cbar_kws = dict(use_gridspec=False,location="top"),vmax=0.30)
    list_ax[0,columns[index]].tick_params(labelsize=16)
    heat.invert_yaxis()

    seaborn.heatmap(round(element[2]['crosstab_vader_stanford']*100,0),linewidths=.5,annot=True, annot_kws={"fontsize":22}, cmap="plasma",ax= list_ax[1,columns[index]],cbar=False, cbar_kws = dict(use_gridspec=False,location="top"))
    seaborn.heatmap(round(element[2]['crosstab_pattern_stanford']*100,0),linewidths=.5,annot=True,cmap = "plasma",annot_kws={"fontsize":22},ax = list_ax[2,columns[index]],cbar=False)
    list_ax[1,columns[index]].tick_params(labelsize=20)
    list_ax[2,columns[index]].tick_params(labelsize=20)

    x_axes = ['Pattern','Vader','Pattern']
    y_axes = ['Vader','Stanford','Stanford']
    for graph in range(3):
        list_ax[graph,columns[index]].set_xlabel(x_axes[graph],fontsize=19, fontweight='bold')
        list_ax[graph,columns[index]].set_ylabel(y_axes[graph], fontsize = 19, fontweight = 'bold')
        if graph != 0:
            list_ax[graph,columns[index]].set_yticklabels(labels=['negative','neutral','positive'],va="center")

plt.figtext(0.273, 0.64, str("r= "+ str(cor_tweets)), ha="left", va="top",fontsize=21, fontweight="bold", color="black")
plt.figtext(0.73, 0.64, str("r= "+ str(cor_speeches)), ha="left", va="top",fontsize=21, fontweight="bold", color="black")

#columns 1 and 3: pie charts#
columns = [1,3]
index1 = 0
for data in enumerate(dataframes):
    list_pie_perc = list(['pie_vader_pattern','pie_vader_stanford','pie_pattern_stanford'])
    y_coord = [0.62,0.35,0.06]
    x_coord = [0.3,0.75]
    dict_pies = data[1][1]
    index2 = 0
    for key,value in dict_pies.items():
        colors = ['seagreen','#ff9999']
        labels = ['agreement','disagreement']
        explode = (0.05, 0.05)
        list_ax[index2,columns[index1]].pie(list(value), colors=colors, autopct='%1.1f%%', startangle=0, pctdistance=0.75,radius=1.3,textprops={'fontsize': 22,'fontweight':'bold'},explode=explode)
        centre_circle = plt.Circle((0, 0), 0.70, fc='#E6E6E6')
        list_ax[index2,columns[index1]].add_artist(centre_circle)
        list_ax[index2,columns[index1]].axis('equal')
        list_ax[index2,columns[index1]].set_position([x_coord[index1], y_coord[index2], 0.18, 0.18]) #left,bottom,width,height
        if (index2 == 0 and index1 == 0) :
            list_ax[index2,columns[index1]].legend(labels,loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10), fontsize= 19,frameon= False)
        index2 = index2 + 1
    index1 = index1 + 1

plt.figtext(0.10, 0.86, "vader vs pattern", ha="center", va="top",fontsize=26, fontweight="bold", color="firebrick")
plt.figtext(0.10, 0.58, "vader vs stanford", ha="center", va="top",fontsize=26,fontweight="bold", color="firebrick")
plt.figtext(0.10, 0.31, "pattern vs stanford", ha="center", va="top",fontsize=26,fontweight="bold", color="firebrick")
fig.add_artist(plt.Line2D([0.5,0.5],[0, 0.8], transform=fig.transFigure, color="black", linewidth = 0.7))

tweeticon = image.imread('tweet icon.png')
image_axis_tweet = fig.add_axes([0.22, 0.855, 0.05, 0.07], zorder=10, anchor="N")
image_axis_tweet.imshow(tweeticon)
image_axis_tweet.axis('off')
plt.figtext(0.28, 0.90, "tweets politicians", ha="left", va="top",fontsize=21, fontweight="bold", color="dimgray")

speechicon = image.imread('speech icon.png')
image_axis_speech= fig.add_axes([0.70, 0.85, 0.05, 0.07], zorder=10, anchor="N")
image_axis_speech.imshow(speechicon)
image_axis_speech.axis('off')
plt.figtext(0.755, 0.90, "presidential speeches", ha="left", va="top",fontsize=21, fontweight="bold", color="dimgray")


plt.figtext(0.05, 0.01, "Sample: 11.467 tweets from Donald Trump, Ted Cruz, Rand Paul, Alexandria Ocasio-Cortez,\nBernie Sanders & Nancy Pelosi. Retweets excluded, duplicates removed.", ha="left", va="bottom",fontsize=18, fontweight="bold", color="dimgray")
plt.figtext(0.52, 0.01, "Sample: 21.943 sentences from 116 presidential speeches, including all inauguration speeches\nand State of the Union addresses", ha="left", va="bottom",fontsize=18, fontweight="bold", color="dimgray")

plt.savefig("agreement_vs_disagreement_libraries.png")
plt.clf()

#see development of sentiment in inauguration speeches#
comparative_graph(sentiment_inaug_kennedy,"sentiment_inauguration MA5.png","Sentiment inauguration speeches US presidents (1961-2017)")

#see evolution throughout time#
mean_sent_speech = sentiment_inaug.groupby('id').agg({'sentiment_stanford': ['mean','std'],'Party': ['first'],'date': ['min']})
mean_sent_speech.columns = ['vader_sent_mean','vader_sent_std','party','date']

x = mean_sent_speech['date']
y = mean_sent_speech['vader_sent_mean']
error = mean_sent_speech['vader_sent_std']

plt.clf()
plt.plot(x,y,'k-')
plt.fill_between(x, y-error, y+error)
plt.savefig("evolution.png")
plt.clf()

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
print("The Mann-Whitney U test statistic is: ",u_statistic)
print ('The P value is: ',pVal)


#evaluated chunks by Vader/Pattern over time#
