# News Article Summariser
Web Scraping Tool for Extracting News Headlines & Summariser

### Description
This Python script is designed for web scraping news headlines from the Today Online website. It utilizes the BeautifulSoup and Selenium libraries to extract data from the HTML content of the specified URL.

### Prerequisites
Make sure you have the following installed before running the script:

    Python (3.12 recommended)
    Required Python packages as listed in the requirements.txt
    Download the required geckodriver for your OS from https://github.com/mozilla/geckodriver/releases

You can install the required packages using the following command:

```
pip install -r requirements.txt
```
### Usage
```
python main.py
```

### Configuration
The script is currently set to run in headless mode using Firefox. If needed, you can customize the web driver options or use a different browser by modifying the script.

```
firefox_options = webdriver.FirefoxOptions()
firefox_options.add_argument("--headless")
browser = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
```


Add this line of code to virtual environment activate file to not get the API limit exceed error.
Replace it with your own Github Personal Access Token

```
export GH_TOKEN = "asdasdasdasd"
```

### License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    Beautiful Soup
    Selenium
    Webdriver Manager