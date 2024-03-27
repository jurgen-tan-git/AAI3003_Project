"""This script scrapes articles from the URLs in the 'URLs' folder and saves
them as text files in the 'articles' folder.
"""

import argparse
import logging
import os
import re
import math
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


def get_html_content(url: str) -> str:
    """Get the HTML content of a webpage using Selenium WebDriver.

    :param url: The URL of the webpage.
    :type url: str
    :return: The HTML content of the webpage.
    :rtype: str
    """
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


def extract_title(html: str) -> str | None:
    """Extract the title of an article from its HTML content.

    :param html: The HTML content of the article.
    :type html: str
    :return: The title of the article, or None if not found.
    :rtype: str | None
    """
    page_soup = BeautifulSoup(html, "html5lib")
    title_element = page_soup.find("h1", class_="h1--page-title")
    if title_element:
        title = title_element.text
        logging.info("Extracted title: %s", title)
        title = re.sub(r"[^a-zA-Z0-9_]", " ", title)
        return "_".join(title.split()).strip()
    logging.warning("No title element found.")
    return None


def extract_article_content(html: str) -> list[str]:
    """Extract the content of an article from its HTML content.

    :param html: The HTML content of the article.
    :type html: str
    :return: The content of the article as a list of paragraphs.
    :rtype: list[str]
    """
    page_soup = BeautifulSoup(html, "html5lib")
    article_content = page_soup.find_all("div", class_="text-long")

    content_list = []
    for content in article_content:
        for p in content.find_all("p"):
            if p.text != "":
                content_list.append(p.text)

    return content_list


def save_to_file(title: str, content_list: list[str], category: str) -> None:
    """Save the article content to a text file.

    :param title: The title of the article.
    :type title: str
    :param content_list: The content of the article as a list of paragraphs.
    :type content_list: list[str]
    :param category: The category of the article.
    :type category: str
    """
    with open(f"{ARTICLES_DIR}/{category}/{title}.txt", "w", encoding="utf-8") as f:
        for content in content_list:
            f.write(content + "\n")


def scrape_article(url: str, category: str) -> None:
    """Scrape an article and save its content to a text file.

    :param url: The URL of the article.
    :type url: str
    :param category: The category of the article.
    :type category: str
    """
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


def scrape_article_multiprocessing_safe(
    url: str, category: str
) -> tuple[str, str, str] | None:
    """Scrape an article and return its title, HTML content, and category. This function is multiprocessing-safe.

    :param url: URL of the article.
    :type url: str
    :param category: Category of the article.
    :type category: str
    :return: A tuple containing the title, HTML content, and category of the article, or None if an error occurred.
    :rtype: tuple[str, str, str] | None
    """
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


def extract_and_save_article(title: str, html: str, category: str) -> None:
    """Extract and save the article content to a text file.

    :param title: The title of the article.
    :type title: str
    :param html: The HTML content of the article.
    :type html: str
    :param category: The category of the article.
    :type category: str
    """
    try:
        content_list = extract_article_content(html)
        if not content_list:
            logging.warning("No article content found.")
        save_to_file(title, content_list, category)
        logging.info("Article successfully scraped and saved.")

    except Exception as e:
        logging.error("An error occurred: %s", str(e))


def main(do_multiprocess: bool = False):
    """The main function to scrape articles from the URLs.

    :param do_multiprocess: Enables multiprocess scraping, defaults to False
    :type do_multiprocess: bool, optional
    """
    categories = os.listdir("./URLs")
    for category in tqdm(categories, position=0, leave=True):
        logging.info("Processing category: %s", category)

        # Create folder to store the articles by categories
        if not os.path.exists(f"{ARTICLES_DIR}/{category[:-4]}"):
            os.makedirs(f"{ARTICLES_DIR}/{category[:-4]}")

        with open(f"URLs/{category}", "r", encoding="utf-8") as f:
            if do_multiprocess:
                # Use 3/4 of the available CPU cores for multiprocessing
                with Pool(int(math.floor(multiprocessing.cpu_count() / 4 * 3))) as p:
                    args = [
                        (url.strip(), category[:-4])
                        for url in tqdm(
                            f,
                            desc="Processing URLs",
                            unit="URL",
                            position=1,
                            leave=False,
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
