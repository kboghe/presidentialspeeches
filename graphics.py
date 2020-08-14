###### IMPORT  #######
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re
import numpy as np
import math
import pandas as pd

#############################################
def mergeDict(dict1, dict2):
  dict3 = {**dict1, **dict2}
  for key, value in dict3.items():
    if key in dict1 and key in dict2:
      dict3[key] = [value, dict1[key]]
  return dict3

#############################################

def comparative_graph(targetfile,name_graph,title_graph):
  ####use the fivethirtyeight style####
  plt.style.use('fivethirtyeight')

  ####create list of possible graphs (max.100)####
  list_ax = list()
  for number in range(100):
    list_ax.append(str("ax"+str(number)))

  ####create plot with subplots####
  rowslen = math.ceil(len(targetfile['id'].drop_duplicates())/3)
  fig, (list_ax) = plt.subplots(rowslen, 3, figsize=((rowslen*3),25), sharey = True) #define rows and columns
  fig.suptitle(title_graph,fontsize=25, fontweight='bold') #define general title of plot

  ####create dict of x values####
  dict_percentages = dict()
  for group in targetfile.groupby("id",sort=False)['percentage']:
    key = group[0]
    dict_percentages[key] = group[1]

  ####create dict of y values####
  dict_values = dict()
  for group in targetfile.groupby("id",sort=False)['sentiment_pattern']:
    key = group[0]
    dict_values.setdefault(key, [])
    dict_values[key].append(group[1].rolling(window=5).mean())
    dict_values[key].append(group[1])

  #####merge both dicts#####
  totaldict = mergeDict(dict_percentages,dict_values)

  #####add metadata#####
  metadata = targetfile[['title','date','President','Party']].drop_duplicates().reset_index(drop=True).to_dict('index')

  #####add means#####
  summary_sentiments = targetfile.groupby('id').agg({'sentiment_vader': ['mean'],'sentiment_pattern': ['mean'],'sentiment_stanford': ['mean']})

  ######add data to plots#######
  row = 0
  column = 0
  graphnum = 0
  for key,value in totaldict.items():

    if row < 4:
      list_ax[row, column].xaxis.set_ticklabels([])

    list_ax[row,column].plot(value[1],value[0][0],linewidth=3.5)[0]
    list_ax[row,column].scatter(value[1],value[0][1],s=6,alpha=.95, c="black")
    list_ax[row,column].axhline(y=0, c="black", linewidth=1.5, zorder=0, linestyle= '--')
    list_ax[row,column].axhline(y=summary_sentiments.iloc[graphnum,1],linewidth = 2.8, zorder=0,linestyle=':')
    column = column + 1
    graphnum = graphnum + 1
    if column > 2:
      column = 0
      row = row+1

  #####add title to plots and match line colors with political party#####
  row = 0
  column = 0
  for value in metadata.values():
    list_ax[row,column].set_title(str(value['President'] + " - " + (str(re.search('[0-9]{4}',value['date']).group(0)))),fontsize=17)
    color_line = np.where(value['Party'] == "Republican","lightcoral","royalblue")
    for i in list([0,2]):
      list_ax[row,column].get_lines()[i].set_color(str(color_line))
      list_ax[row,column].get_lines()[i].set_label(str(value['Party']))
    column = column + 1
    if column > 2:
      column = 0
      row = row+1

  #####add legend#####
  colors = ['lightcoral','royalblue','black']
  linestyles = ['solid','solid',':']
  lines = [Line2D([0], [0], color=c, linewidth=3, linestyle= l) for (c,l) in zip(colors,linestyles)]
  labels = ['Republican', 'Democrat','mean sentiment']
  fig.legend(lines, labels, loc="center", bbox_to_anchor=(0.5, 0.93),ncol=3,frameon=False,fontsize=16)

  #####add general X- and Y-axis labels######
  fig.text(0.5, 0.04, 'Speech progression (0% (begin) - 100% (end))', ha='center', va='center',fontsize=19)
  fig.text(0.01, 0.5, 'Sentiment (< 0: negative, > 0: positive)', ha='left', va='center', rotation = 'vertical',fontsize=19)

  ######save figure#######
  plt.savefig(name_graph)

  ######clear the current figure######
  fig.clf()

###########################

def comparative_figures(data):
  dict_cat = dict()
  variables = ['sentiment_vader', 'sentiment_pattern', 'sentiment_stanford']
  categories = ['cat_' + element for element in variables]
  bins_cut = [[-1, -0.00001, 0.00001, 1], [-1, -0.00001, 0.00001, 1], [-10, 1.99, 2.01, 10]]
  labels_cut = ['negative', 'neutral', 'positive']
  for index, category in enumerate(categories):
    dict_cat[category] = pd.cut(data[variables[index]], bins=bins_cut[index], include_lowest=True,labels=labels_cut)

  dict_compare = dict()
  comparisons = ['vader_pattern', 'vader_stanford', 'pattern_stanford']
  categories = ['disagreement_' + element for element in comparisons]
  compare1 = ['cat_sentiment_' + element for element in ['vader','vader','pattern']]
  compare2 = ['cat_sentiment_' + element for element in ['pattern','stanford','stanford']]
  for index, category in enumerate(categories):
    dict_compare[category] = np.where(dict_cat[compare1[index]] != dict_cat[compare2[index]],"disagreement","agreement")
  comparison_pd = {**dict_cat, **dict_compare}
  comparison_pd = pd.DataFrame.from_dict(comparison_pd)

  dict_pies = dict()
  pies = ['pie_' + element for element in comparisons]
  for index, category in enumerate(categories):
    dict_pies[pies[index]] = comparison_pd.groupby(category)[category].count() / len(data) * 100

  dict_crosstabs = dict()
  crosstabs = ['crosstab_' + element for element in comparisons]
  heatmap = data[['sentiment_vader', 'sentiment_pattern']]
  heatmap = heatmap[~(heatmap == 0).any(axis=1)]
  dict_crosstabs[crosstabs[0]] = pd.crosstab(round(heatmap['sentiment_vader'], 1), round(heatmap['sentiment_pattern'], 1), normalize='index')
  dict_crosstabs[crosstabs[1]] = pd.crosstab(comparison_pd['cat_sentiment_stanford'], comparison_pd['cat_sentiment_vader'],normalize='index')
  dict_crosstabs[crosstabs[2]] = pd.crosstab(comparison_pd['cat_sentiment_stanford'], comparison_pd['cat_sentiment_pattern'],normalize='index')

  return [comparison_pd, dict_pies, dict_crosstabs]

##### correlation and type of disagreement #######

def corr_info(data):
  heatmap = data[['sentiment_vader', 'sentiment_pattern']]
  heatmap = heatmap[~(heatmap == 0).any(axis=1)]
  corr = heatmap.corr()
  variables = ['sentiment_vader', 'sentiment_pattern']
  categories = ['cat_' + element for element in variables]
  bins_cut = [[-1, -0.00001, 0.00001, 1], [-1, -0.00001, 0.00001, 1]]
  labels_cut = ['negative', 'neutral', 'positive']
  for index, category in enumerate(categories):
    data[category] = pd.cut(data[variables[index]], bins=bins_cut[index], include_lowest=True,labels=labels_cut)
  data['disagreement'] = np.where(data['cat_sentiment_vader'] != data['cat_sentiment_pattern'],"disagreement","agreement")
  data['type_disagreement'] = np.where(((data['sentiment_vader'] == 0) | (data['sentiment_pattern'] == 0)),"one_neutral","other")
  disagreement_type = pd.crosstab(data['disagreement'],data['type_disagreement'],normalize='index')
  return [corr,disagreement_type]


