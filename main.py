from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support import expected_conditions as EC
import re
import logging
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def get_html_content(url):
    browser = None
    try:
        # Set up Firefox WebDriver
        firefox_options = webdriver.FirefoxOptions()
        firefox_options.add_argument("--headless")
        browser = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)

        # Load the URL and wait for elements to be present
        browser.get(url)
        WebDriverWait(browser, 3).until(EC.presence_of_all_elements_located)

        # Extract HTML content
        return browser.page_source
    
    finally:
        # Close the browser
        if browser:
            browser.quit()

def extract_title(html):
    page_soup = BeautifulSoup(html, 'html5lib')
    title_element = page_soup.find("h1", class_="h1--page-title")
    if title_element:
        title = title_element.text
        logging.info(f"Extracted title: {title}")
        title = re.sub(r'[^a-zA-Z0-9_]', ' ', title)
        return '_'.join(title.split()).strip()
    else:
        logging.warning("No title element found.")
        return None

def extract_article_content(html):
    page_soup = BeautifulSoup(html, 'html5lib')
    article_content = page_soup.find_all("div", class_="text-long")

    content_list = []
    for content in article_content:
        for p in content.find_all('p'):
            if p.text != '':
                content_list.append(p.text)

    return content_list

def save_to_file(title, content_list):
    with open(f'articles/{title}.txt', 'w', encoding='utf-8') as f:
        for content in content_list:
            f.write(content + "\n")

def scrape_article(url):
    # Create folder to store the articles
    if not os.path.exists('articles'):
        os.makedirs('articles')

    try:
        html = get_html_content(url)
        if not html:
            logging.error("Failed to retrieve HTML content.")
            return

        title = extract_title(html)
        if not title:
            logging.error("Failed to extract title.")
            return

        content_list = extract_article_content(html)
        if not content_list:
            logging.warning("No article content found.")

        save_to_file(title, content_list)
        logging.info("Article successfully scraped and saved.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    with open('todayonline.txt', 'r') as f:
        for url in tqdm(f, desc="Processing URLs", unit="URL"):
            try:
                url = url.strip()
                logging.info(f"Processing URL: {url}")
                scrape_article(url)
            except Exception as e:
                logging.error(f"Error processing URL '{url}': {e}")

    logging.info("Processing complete.")
