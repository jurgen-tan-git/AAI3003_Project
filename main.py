from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager



url = 'https://www.todayonline.com'

firefox_options = webdriver.FirefoxOptions()
firefox_options.add_argument("--headless")
browser = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)

browser.get(url)
element = WebDriverWait(browser, 3)
html = browser.page_source
page_soup = BeautifulSoup(html, 'html5lib')

coverpage_news = page_soup.find_all("a", class_="list-object__heading-link")
# coverpage_news = page_soup.find_all("h6", class_="list-object__heading")
for news in coverpage_news:
    # print(news)
    news_soap = BeautifulSoup(str(news), 'html.parser')
    for a in news_soap.find_all('a', href=True): 
        print(a['href'])