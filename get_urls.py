import argparse
import logging
import os

from bs4 import BeautifulSoup, ResultSet
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.firefox import GeckoDriverManager
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)


def scrape_today_online(url, outputfile):
    # Set up Firefox WebDriver
    firefox_options = webdriver.FirefoxOptions()
    firefox_options.add_argument("--headless")
    browser = webdriver.Firefox(
        service=FirefoxService(GeckoDriverManager().install()), options=firefox_options
    )

    try:
        # Load the URL and wait for elements to be present
        browser.get(url)
        WebDriverWait(browser, 3).until(EC.presence_of_all_elements_located)

        # Extract HTML content
        html = browser.page_source
        page_soup = BeautifulSoup(html, "html5lib")

        # Extract news headlines and write to a file
        coverpage_news = page_soup.find_all("a", class_="list-object__heading-link")
        with open(outputfile, "w", encoding="utf-8") as f:
            for news in tqdm(coverpage_news):
                news_soap = BeautifulSoup(str(news), "html.parser")
                for a in news_soap.find_all("a", href=True):
                    if a["href"].startswith("https"):
                        f.write(a["href"] + "\n")
                    else:
                        f.write(url + a["href"] + "\n")

    finally:
        # Close the browser
        browser.quit()


def main():

    categories = [
        "singapore",
        "world",
        "big-read",
        "adulting-101",
        "gen-y-speaks",
        "gen-z-speaks",
        "voices",
        "commentary",
    ]

    # Create folder to store the output files
    if not os.path.exists("URLs"):
        os.makedirs("URLs")

    for category in categories:
        url_to_scrape = "https://www.todayonline.com"
        output_filename = "./URLs/" + category + ".txt"
        scrape_today_online(url_to_scrape, output_filename)


if __name__ == "__main__":
    main()
