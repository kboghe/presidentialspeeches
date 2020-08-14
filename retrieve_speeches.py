from bs4 import BeautifulSoup as bs
import pandas as pd
import time
from selenium import webdriver
import re
import dateparser

######################################
#1. scrape speeches from MillerCenter#
######################################

#start Chromedriver and access MillerCenter website#
driver = webdriver.Chrome('./chromedriver')
url = "https://millercenter.org/the-presidency/presidential-speeches"
driver.get(url)
driver.implicitly_wait(10)

#keep scrolling down until page stops loading additional records#
pause_scroll = 6
last_try = 0
initialcoord = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(pause_scroll)
    newcoord = driver.execute_script("return document.body.scrollHeight")
    if newcoord == initialcoord:
        time.sleep(20)
        last_try = last_try + 1
    if last_try > 1:
        break
    newcoord = initialcoord

#retrieve urls to all speeches#
page_source = driver.page_source
bsobject_linkpage = bs(page_source,'lxml')
links = bsobject_linkpage.find_all("a", href= re.compile('presidential-speeches/'))
link_list = list()
for link in links:
    link_specific = link['href']
    link_list.append("https://millercenter.org"+link_specific)

link_list.append("https://millercenter.org/the-presidency/presidential-speeches/january-8-2020-statement-iran")
link_list.append("https://millercenter.org/the-presidency/presidential-speeches/january-3-2020-remarks-killing-qasem-soleimani")
link_list.append("https://millercenter.org/the-presidency/presidential-speeches/october-27-2019-statement-death-abu-bakr-al-baghdadi")
link_list.append("https://millercenter.org/the-presidency/presidential-speeches/september-25-2019-press-conference")
link_list.append("https://millercenter.org/the-presidency/presidential-speeches/september-24-2019-remarks-united-nations-general-assembly")

#scrape the speech#
title, speech, name, date, about = ([] for i in range(5))
for index,link in enumerate(link_list):
    #access speech page with Selenium and load html source into Beautifulsoup#
    driver.get(link_list[index])
    driver.find_elements_by_css_selector('div[class="transcript-inner"]')
    page_source = driver.page_source
    bsobject_speechpage = bs(page_source, 'lxml')

    #scrape speech and other properties#
    title.append((bsobject_speechpage.find('h2', class_="presidential-speeches--title").text).rstrip())
    try:
        speech_raw = bsobject_speechpage.find('div', class_="transcript-inner").text
    except:
        speech_raw = (bsobject_speechpage.find('div', class_="view-transcript").text).rstrip()
    speech.append(re.sub("Transcript|\\n"," ",speech_raw))
    name.append((bsobject_speechpage.find('p', class_="president-name").text).rstrip())
    date.append((dateparser.parse(bsobject_speechpage.find('p', class_="episode-date").text)))
    try:
        about_raw = bsobject_speechpage.find('div', class_="about-sidebar--intro").text.rstrip()
    except:
        about_raw = about.append("No info available")
    empty = ""
    if about_raw == empty:
        about_raw = "No info available"
    about.append(re.sub("\\n"," ",about_raw))

#save this to a dataframe and save to a csv file#
speeches_presidents = pd.DataFrame({'name':name,'title':title,'date':date,'info':about,'speech':speech}, columns=['name','title','date','info','speech'])
speeches_presidents['speech'] = speeches_presidents['speech'].apply(lambda x: x.replace(".",". "))
speeches_presidents.to_csv("speeches.csv", encoding="utf-8",quotechar="'",index=False)

###################################################
#2. scrape extra info on presidents from Wikipedia#
###################################################

#scrape info from presidents#
url = "https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States"
tables_wiki = pd.read_html(url)[1]
lastrow = int(len(tables_wiki)-1)
tables_wiki = tables_wiki.drop([lastrow])
tables_wiki = tables_wiki.replace("\[(.*)\]","", regex=True)
tables_wiki = tables_wiki.drop(columns=['Presidency[a]', 'President','Party[b]'])
tables_wiki.columns = tables_wiki.columns.str.replace("\[(.*)\]","", regex=True)
tables_wiki.columns = tables_wiki.columns.str.replace("\.(.*)","", regex=True)
tables_wiki['from'] = tables_wiki['Presidency'].apply(lambda x: dateparser.parse(x.split("–")[0]).year)
tables_wiki['until'] = tables_wiki['Presidency'].apply(lambda x: dateparser.parse(x.split("–")[1]).year if x.split("–")[1] != "Incumbent" else "2021")
tables_wiki = tables_wiki.drop(columns=['Presidency'])

#scrape info from VP's#
tables_wiki_vp = tables_wiki[~tables_wiki['Vice President'].str.contains("Vacant")]
number_vice_presidents = pd.DataFrame(tables_wiki_vp.groupby("President")['Vice President'].nunique()).reset_index()
tables_wiki = tables_wiki.drop(columns=['Vice President','Election'])
tables_wiki = tables_wiki.merge(number_vice_presidents, on="President", how='left')

#change and replace several values to ensure consistency with data from the Miller Center#
tables_wiki = tables_wiki.replace("Richard Nixon","Richard M. Nixon", regex=True)
tables_wiki = tables_wiki.replace("William Howard Taft","William Taft", regex=True)
tables_wiki = tables_wiki.replace("William Henry Harrison","William Harrison", regex=True)
tables_wiki['President'].loc[(tables_wiki['President'] == "Grover Cleveland") & (tables_wiki['from'] == 1885)] = "Grover Cleveland 1"
tables_wiki['President'].loc[(tables_wiki['President'] == "Grover Cleveland") & (tables_wiki['from'] == 1893)] = "Grover Cleveland 2"
tables_wiki = tables_wiki.drop_duplicates().reset_index(drop=True)
tables_wiki = tables_wiki.drop([16]).reset_index(drop=True)
tables_wiki['Vice President'] = tables_wiki['Vice President'].fillna("0")

################################################################
#3. load the speeches into Python again and merge with metadata#
################################################################

#perform several changes to the speeches dataset before merging#
presidential_speeches = pd.read_csv("speeches.csv", encoding="utf-8",quotechar="'")
presidential_speeches['date'] = presidential_speeches['date'].apply(lambda x: dateparser.parse(x))
presidential_speeches = presidential_speeches.rename(columns={'name':"President"})
presidential_speeches['President'].loc[(presidential_speeches['President'] == "Grover Cleveland") & (presidential_speeches['date'] < dateparser.parse("1893-01-01"))] = "Grover Cleveland 1"
presidential_speeches['President'].loc[(presidential_speeches['President'] == "Grover Cleveland") & (presidential_speeches['date'] > dateparser.parse("1893-01-01"))] = "Grover Cleveland 2"
presidential_speeches['info'] = presidential_speeches['info'].fillna("No info available")

#join the data and write to hard drive#
presidential_speeches = presidential_speeches.merge(tables_wiki, on="President", how='left')
presidential_speeches['speech'] = presidential_speeches['speech'].apply(lambda x: x.replace(".",". "))
presidential_speeches = presidential_speeches[['President', 'Party', 'from','until', 'Vice President', 'title', 'date', 'info', 'speech']]
presidential_speeches.to_csv("2presidential_speeches_with_metadata.csv", sep = ";", encoding="utf-8",quotechar="'",index=False)