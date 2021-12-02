from bs4 import BeautifulSoup
from selenium import webdriver
# 避免网页封ip，模拟chrome浏览器，进行网页的爬取
chrome_options = webdriver.ChromeOptions()
chrome_options.headless = True
chrome = webdriver.Chrome(chrome_options=chrome_options)

url = ''
page = chrome.get(url)

# url : https://www.zillow.com/standford-ca/sold
# url : https://www.zillow.com/standford-ca/sold/2-p
# ...

# get house id from index pages
html_path=url
page = BeautifulSoup(open(html_path, 'r'))
links = [a['herf'] for a in page.find_all('a', 'list-card-link')]
ids = [l.split('/')[-2].split('_')[0] for l in links]

# so that can query detail page by id

# extract data
sold_items = [a.text for a in page.find('div', 'ds-home-details-chip').find('p').find_all('span')]

for item in sold_items:
    if 'Sold:' in item:
        result['Sold Price'] = item.split(' ')[1]
    if 'Sold on' in item:
        result['Sold On'] = item.split(' ')[-1]

