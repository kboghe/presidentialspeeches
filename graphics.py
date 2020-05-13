import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import re

def comparative_graph(targetfile):

  ####define style use####
  plt.style.use('fivethirtyeight')

  ####create list of possible graphs (max.100)####
  list_ax = list()
  for number in range(100):
    list_ax.append(str("ax"+str(number)))

  ####create plot with subplots####
  fig, (list_ax) = plt.subplots(5, 3, figsize=(15,25), sharey = True) #define rows and columns
  fig.suptitle('Sentiment of inaugural speeches US presidents since John F. Kennedy',fontsize=25, fontweight='bold') #define general title of plot


  dict_percentages = dict()
  for group in targetfile.groupby("id",sort=False)['percentage']:
    key = group[0]
    dict_percentages[key] = group[1]


  dict_values = dict()
  for group in targetfile.groupby("id",sort=False)['sentiment_vader']:
    key = group[0]
    dict_values.setdefault(key, [])
    dict_values[key].append(group[1].rolling(window=10).mean())
    dict_values[key].append(group[1])


  metadata = targetfile[['title','date','President','Party']].drop_duplicates().reset_index(drop=True).to_dict('index')

  def mergeDict(dict1, dict2):
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
      if key in dict1 and key in dict2:
        dict3[key] = [value, dict1[key]]
    return dict3

  totaldict = mergeDict(dict_percentages,dict_values)

  row = 0
  column = 0
  for key,value in totaldict.items():
    list_ax[row,column].plot(value[1],value[0][0],linewidth=3)[0]
    list_ax[row,column].scatter(value[1],value[0][1],s=6,alpha=.95, c="black")
    list_ax[row,column].axhline(y=0, c="black", linewidth=1.2, zorder=0, linestyle= '--')
    column = column + 1
    if column > 2:
      column = 0
      row = row+1

  row = 0
  column = 0
  for value in metadata.values():
    list_ax[row,column].set_title(str(value['President'] + " - " + (str(re.search('[0-9]{4}',value['date']).group(0)))),fontsize=17)
    color_line = np.where(value['Party'] == "Republican","lightcoral","royalblue")
    list_ax[row,column].get_lines()[0].set_color(str(color_line))
    list_ax[row,column].get_lines()[0].set_label(str(value['Party']))
    column = column + 1
    if column > 2:
      column = 0
      row = row+1

  colors = ['lightcoral','royalblue']
  lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
  labels = ['Republican', 'Democrat']
  fig.legend(lines, labels, loc="center", bbox_to_anchor=(0.5, 0.93),ncol=2,frameon=False,fontsize=20)

  fig.text(0.5, 0.04, 'Speech progression (0% (begin) - 100% (end))', ha='center', va='center',fontsize=19)
  fig.text(0.01, 0.5, 'Sentiment (< 0: negative, > 0: positive)', ha='left', va='center', rotation = 'vertical',fontsize=19)

  plt.savefig('graph-fivethirty.png')

  fig.clf()