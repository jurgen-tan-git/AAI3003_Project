import argparse
import logging
import os
import re
import multiprocessing

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from webdriver_manager.firefox import GeckoDriverManager
from multiprocess import Pool

logging.basicConfig(level=logging.INFO)
ARTICLES_DIR = "./articles"


def get_html_content(url):
    browser = None
    try:
        # Set up Firefox WebDriver
        firefox_options = webdriver.FirefoxOptions()
        firefox_options.add_argument("--headless")
        browser = webdriver.Firefox(
            service=FirefoxService(GeckoDriverManager().install()),
            options=firefox_options,
        )

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
    page_soup = BeautifulSoup(html, "html5lib")
    title_element = page_soup.find("h1", class_="h1--page-title")
    if title_element:
        title = title_element.text
        logging.info("Extracted title: %s", title)
        title = re.sub(r"[^a-zA-Z0-9_]", " ", title)
        return "_".join(title.split()).strip()
    logging.warning("No title element found.")
    return None


def extract_article_content(html):
    page_soup = BeautifulSoup(html, "html5lib")
    article_content = page_soup.find_all("div", class_="text-long")

    content_list = []
    for content in article_content:
        for p in content.find_all("p"):
            if p.text != "":
                content_list.append(p.text)

    return content_list


def save_to_file(title, content_list, category):
    with open(f"{ARTICLES_DIR}/{category}/{title}.txt", "w", encoding="utf-8") as f:
        for content in content_list:
            f.write(content + "\n")


def scrape_article(url, category):
    # Create folder to store the articles
    if not os.path.exists(ARTICLES_DIR):
        os.makedirs(ARTICLES_DIR)

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
        save_to_file(title, content_list, category)
        logging.info("Article successfully scraped and saved.")

    except Exception as e:
        logging.error("An error occurred: %s", str(e))


def scrape_article_multiprocessing_safe(url, category):
    try:
        html = get_html_content(url)
        if not html:
            logging.error("Failed to retrieve HTML content.")
            return
        title = extract_title(html)
        if not title:
            logging.error("Failed to extract title.")
            return

        return title, html, category

    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        return


def extract_and_save_article(title, html, category):
    try:
        content_list = extract_article_content(html)
        if not content_list:
            logging.warning("No article content found.")
        save_to_file(title, content_list, category)
        logging.info("Article successfully scraped and saved.")

    except Exception as e:
        logging.error("An error occurred: %s", str(e))


def main(do_multiprocess: bool = False):
    categories = os.listdir("./URLs")
    for category in tqdm(categories, position=0, leave=True):
        logging.info("Processing category: %s", category)

        # Create folder to store the articles by categories
        if not os.path.exists(f"{ARTICLES_DIR}/{category[:-4]}"):
            os.makedirs(f"{ARTICLES_DIR}/{category[:-4]}")

        with open(f"URLs/{category}", "r", encoding="utf-8") as f:
            if do_multiprocess:
                with Pool(multiprocessing.cpu_count() // 4 * 3) as p:
                    args = [
                        (url.strip(), category[:-4])
                        for url in tqdm(
                            f, desc="Processing URLs", unit="URL", position=1, leave=False
                        )
                    ]
                    results = p.starmap(scrape_article_multiprocessing_safe, args)
                for result in tqdm(
                    results,
                    desc="Saving articles",
                    unit="article",
                    position=1,
                    leave=False,
                ):
                    if result:
                        title, html, category = result
                        extract_and_save_article(title, html, category)
            else:
                for url in tqdm(
                    f, desc="Processing URLs", unit="URL", position=1, leave=False
                ):
                    try:
                        url = url.strip()
                        scrape_article(url, category[:-4])
                    except Exception as e:
                        logging.error("Error processing URL '%s': %s", url, str(e))
    logging.info("Processing complete.")


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "-m",
        "--multiprocessing",
        help="Use multiprocessing",
        action="store_true",
        dest="do_multiprocess",
    )
    main(**vars(arg.parse_args()))
