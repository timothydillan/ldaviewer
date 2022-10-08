import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import http.client
import urllib.parse
import uuid
import json
import ssl
import re
from lxml import etree
from fake_useragent import UserAgent
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import requests
import pandas as pd
from urllib.parse import urlencode
import re
import arxiv
import datetime
import time
import numpy as np


class GoogleScholar:
    STEP_INCREMENT = 10

    def __init__(self):
        pass

    def get_excerpts_from_keyword(self, keyword):
        options = Options()
        options.headless = True
        ua = UserAgent()
        user_agent = ua.random
        options.add_argument(f'user-agent={user_agent}')
        # options.add_argument("--window-size=1920,1200")
        driver = webdriver.Chrome(service=Service(
            ChromeDriverManager().install()), options=options)
        driver.get(
            f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={keyword}")
        element_present = EC.presence_of_element_located((By.ID, 'gs_ab_md'))
        try:
            WebDriverWait(driver, 10).until(element_present)
        except:
            print("Failed to load google scholar in the predetermined timeout.")
            return

        excerpts = []
        excerpt_1 = self.__get_excerpts_from_source(
            driver.page_source.encode('utf-8'))
        for excerpt in excerpt_1:
            excerpts.append(excerpt)

        while True:
            element_present = EC.presence_of_element_located(
                (By.CLASS_NAME, 'gs_btnPR'))
            try:
                WebDriverWait(driver, 10).until(element_present)
            except:
                print("Failed to load next page.")
                break
            driver.find_element(by=By.CLASS_NAME, value="gs_btnPR").click()
            excerpt = self.__get_excerpts_from_source(
                driver.page_source.encode('utf-8'))
            for excerpt in excerpt_1:
                excerpts.append(excerpt)
            print("sleeping for 2 sec and repeating the process for the next batch..")
            time.sleep(2)

        driver.close()
        return excerpts

    def get_excerpts_from_keyword_new(self, keyword):
        excerpts = []
        step = 0
        while True:
            options = Options()
            options.headless = True
            ua = UserAgent()
            user_agent = ua.random
            options.add_argument(f'user-agent={user_agent}')
            driver = webdriver.Chrome(service=Service(
                ChromeDriverManager().install()), options=options)
            driver.get(
                f"https://scholar.google.com/scholar?hl=en&q={keyword}&start={step}")
            element_present = EC.presence_of_element_located(
                (By.CLASS_NAME, 'gs_btnPR'))
            try:
                WebDriverWait(driver, 10).until(element_present)
            except:
                print("Failed to load google scholar in the predetermined timeout.")
                break
            excerpt = self.__get_excerpts_from_source(
                driver.page_source.encode('utf-8'))
            excerpts.append(excerpt)
            print(excerpt)
            driver.close()
            step += self.STEP_INCREMENT
        return excerpts

    def __get_excerpts_from_source(self, data):
        soup = BeautifulSoup(data, "html.parser")
        excerptsObj = soup.find_all('div', class_=re.compile(".*gs_rs.*"))
        if excerptsObj is None:
            print("Can't find any excerpts.")
            return []
        return [excerpt.text for excerpt in excerptsObj]


class CORE:
    # API-related constants.
    API_HOST = "https://api.core.ac.uk/v3/"
    API_KEY = "vPftOYNzhC2ZjnK5SGcDTEb7mglW18BH"
    SEARCH_API_PATH = "search"
    ENTITY_TYPE_WORK = "works"

    # Response keys
    TOTAL_HITS_KEY = "totalHits"
    RESULTS_KEY = "results"
    ABSTRACT_KEY = "abstract"
    FULL_TEXT_KEY = "fullText"
    CREATED_DATE_KEY = "createdDate"
    PUBLISHED_DATE_KEY = "publishedDate"
    UPDATED_DATE_KEY = "updatedDate"
    FULL_TEXT_URL_KEY = "sourceFulltextUrls"
    TITLE_KEY = "title"
    IDENTIFIER_KEY = "id"

    def __init__(self) -> None:
        pass

    def __is_works_results_valid_works(self):
        if self.works is None:
            print("No works were retrieved!")
            return
        if self.TOTAL_HITS_KEY not in self.works:
            print("Invalid works payload.")
            return False
        if self.works[self.TOTAL_HITS_KEY] <= 0:
            print("No results found.")
            return False
        if self.RESULTS_KEY not in self.works:
            print("No results found.")
            return False
        if len(self.works[self.RESULTS_KEY]) <= 0:
            print("No results found.")
            return False
        return True

    def get_abstracts_from_works(self, save_to_csv=False):
        if self.works is None:
            print("No works were retrieved!")
            return
        if not self.__is_works_results_valid_works():
            return
        abstracts = [result[self.ABSTRACT_KEY]
                     for result in self.works[self.RESULTS_KEY]]
        if save_to_csv:
            pd.DataFrame.from_dict(
                {"abstract": abstracts}).to_csv('abstracts.csv')
        return abstracts

    def get_fulltext_from_works(self, save_to_csv=False):
        if self.works is None:
            print("No works were retrieved!")
            return
        if not self.__is_works_results_valid_works():
            return
        full_texts = [result[self.FULL_TEXT_KEY]
                      for result in self.works[self.RESULTS_KEY]]
        if save_to_csv:
            pd.DataFrame.from_dict(
                {"full_text": full_texts}).to_csv('fulltext.csv')
        return full_texts

    def get_data_from_works(self, save_to_csv=False):
        if self.works is None:
            print("No works were retrieved!")
            return
        if not self.__is_works_results_valid_works():
            return

        data_dict = {
            self.IDENTIFIER_KEY: [],
            self.TITLE_KEY: [],
            self.ABSTRACT_KEY: [],
            self.FULL_TEXT_KEY: [],
            self.CREATED_DATE_KEY: [],
            self.PUBLISHED_DATE_KEY: [],
            self.UPDATED_DATE_KEY: [],
            self.FULL_TEXT_URL_KEY: []
        }

        for result in self.works[self.RESULTS_KEY]:
            for column in data_dict.keys():
                value = eval(f"result['{column}']")
                if value is not None:
                    data_dict[column].append(eval(f"result['{column}']"))

        data_dict[self.ABSTRACT_KEY] = list(
            map(lambda x: x.split(".Comment")[0], data_dict[self.ABSTRACT_KEY]))
        data_dict["published_date"] = data_dict.pop(self.PUBLISHED_DATE_KEY)

        if save_to_csv:
            pd.DataFrame.from_dict(data_dict).to_csv('core_scraping.csv')

        return data_dict

    def get_works_by_search_query(self, query, limit=10):
        if len(query.strip()) == 0:
            print("Error! Queries cannot be empty")
            return

        payload = {
            "q": query,
            "scroll": True,
            "offset": 0,
            "limit": limit,
            "scroll_id": "",
            "stats": True,
            "raw_stats": True,
            "exclude": None,
            "sort": None,
            "accept": "",
            "measure": True
        }

        self.works, sec = self.__post_request(
            f"{self.SEARCH_API_PATH}/{self.ENTITY_TYPE_WORK}", payload)
        print("Took", sec, "seconds to get data.")
        return self.works

    def __post_request(self, path, payload):
        headers = {"Authorization": f"Bearer {self.API_KEY}"}
        response = requests.post(
            f"{self.API_HOST}{path}", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json(), response.elapsed.total_seconds()
        else:
            print(f"Error code {response.status_code}, {response.content}")
            return None, 0


class COREV2:
    # API-related constants.
    API_HOST = "https://api.core.ac.uk/v3/"
    API_KEY = "vPftOYNzhC2ZjnK5SGcDTEb7mglW18BH"
    SEARCH_API_PATH = "search"
    ENTITY_TYPE_WORK = "works"

    # Response keys
    TOTAL_HITS_KEY = "totalHits"
    RESULTS_KEY = "results"
    ABSTRACT_KEY = "abstract"
    FULL_TEXT_KEY = "fullText"

    def __init__(self) -> None:
        pass

    def __is_works_results_valid_works(self):
        if self.works is None:
            print("No works were retrieved!")
            return
        if self.TOTAL_HITS_KEY not in self.works:
            print("Invalid works payload.")
            return False
        if self.works[self.TOTAL_HITS_KEY] <= 0:
            print("No results found.")
            return False
        if self.RESULTS_KEY not in self.works:
            print("No results found.")
            return False
        if len(self.works[self.RESULTS_KEY]) <= 0:
            print("No results found.")
            return False
        return True

    def get_data_from_works(self, save_to_csv=False):
        if self.works is None:
            print("No works were retrieved!")
            return
        if not self.__is_works_results_valid_works():
            return
        abstracts = []
        full_texts = []
        for result in self.works[self.RESULTS_KEY]:
            abstracts.append(result[self.ABSTRACT_KEY])
            full_texts.append(result[self.FULL_TEXT_KEY])
        if save_to_csv:
            pd.DataFrame.from_dict(
                {"abstract": abstracts, "full_texts": full_texts}).to_csv('core_scraping.csv')
        return abstracts

    def get_works_by_search_query(self, query, limit=10):
        if len(query.strip()) == 0:
            print("Error! Queries cannot be empty")
            return

        payload = {
            "q": query,
            "scroll": True,
            "offset": 0,
            "limit": limit,
            "scroll_id": "",
            "stats": True,
            "raw_stats": True,
            "exclude": None,
            "sort": None,
            "accept": "",
            "measure": True
        }

        self.works, sec = self.__post_request(
            f"{self.SEARCH_API_PATH}/{self.ENTITY_TYPE_WORK}", payload)
        print("Took", sec, "seconds to get data.")
        return self.works

    def __post_request(self, path, payload):
        headers = {"Authorization": f"Bearer {self.API_KEY}"}
        response = requests.post(
            f"{self.API_HOST}{path}", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json(), response.elapsed.total_seconds()
        else:
            print(f"Error code {response.status_code}, {response.content}")
            return None, 0


class Emerald:
    # API-related constants.
    HOST = "https://www.emerald.com/"
    SEACRH_PATH = "insight/search"

    # Query constants
    QUERY_KEY = "q"
    ADVANCED_KEY = "advanced"
    FROM_YEAR_KEY = "fromYear"
    TO_YEAR_KEY = "toYear"
    LIMIT_KEY = "ipp"
    PAGE_KEY = "p"

    # Content type constants
    CONTENT_TYPE_KEY = "content-type"
    CONTENT_TYPE_ARTICLE = "article"

    # Fields to search constants
    ABSTRACT_FIELD = "abstract"

    # Important elements to look for
    ABSTRACT_CLASS_NAME = "intent_abstract"
    P_TAG_NAME = "p"

    MAX_ATTEMPT_COUNT = 5

    def __init__(self) -> None:
        pass

    def get_data_from_query(self, search_query):
        if len(search_query) <= 0:
            print("Cannot search with empty query.")
            return

        abstracts = []
        p = 1
        attempt_count = 1
        while True:
            options = Options()
            options.headless = True
            ua = UserAgent()
            user_agent = ua.random
            options.add_argument(f'user-agent={user_agent}')
            driver = webdriver.Chrome(service=Service(
                ChromeDriverManager().install()), options=options)

            payload = {
                self.QUERY_KEY: f"({self.CONTENT_TYPE_KEY}:{self.CONTENT_TYPE_ARTICLE}) AND {search_query}",
                self.ADVANCED_KEY: True,
                self.LIMIT_KEY: 100,
            }

            if p > 1:
                payload[self.PAGE_KEY] = p

            url = f"{self.HOST}{self.SEACRH_PATH}?{urlencode(payload)}"
            print(url)

            driver.get(url)
            driver.implicitly_wait(10)
            element_present = EC.presence_of_element_located(
                (By.CLASS_NAME, 'intent_title'))
            try:
                WebDriverWait(driver, 10).until(element_present)
            except:
                print("Failed to load articles in the predetermined timeout.")
                driver.close()
                break

            abstract = self.__get_abstracts_from_source(
                driver.page_source.encode('utf-8'))
            abstracts.append(abstract)

            if attempt_count >= self.MAX_ATTEMPT_COUNT:
                print("Reached max attempt count of",
                      self.MAX_ATTEMPT_COUNT, ". Stopping scraping.")
                driver.close()
                break

            next_element_present = EC.presence_of_element_located(
                (By.CLASS_NAME, 'intent_next_page_link'))
            try:
                WebDriverWait(driver, 10).until(next_element_present)
            except:
                print("No next page found. Stopping scraping.")
                driver.close()
                break

            driver.close()

            attempt_count += 1
            p += 1

        return abstracts

    def __get_abstracts_from_source(self, data):
        soup = BeautifulSoup(data, "html.parser")
        abstractsObj = soup.find_all(
            'div', class_=re.compile(f".*{self.ABSTRACT_CLASS_NAME}.*"))
        if abstractsObj is None:
            print("Can't find any abstracts.")
            return []

        abstracts = []
        for abstractObjs2 in abstractsObj:
            abstract = ""
            for p in abstractObjs2.find_all(f"{self.P_TAG_NAME}"):
                abstract += p.text
            # Remove extra spaces
            abstract = re.sub("  +", " ", abstract)
            abstracts.append(abstract)

        return abstracts


class Arxiv:
    FILTER_BY_DATE_LIMIT = 100

    def __init__(self) -> None:
        pass

    def __is_in_date_range(self, article_timestamp, start_date, end_date):
        return start_date <= article_timestamp.replace(tzinfo=None) <= end_date

    def get_data_from_query(self, search_query, sort_by=arxiv.SortCriterion.Relevance, limit=50, save_to_csv=False, filter_by_date=False, start_date=None, end_date=None):
        if len(search_query) <= 0:
            print("Cannot search with empty query.")
            return

        if filter_by_date:
            if sort_by not in [arxiv.SortCriterion.SubmittedDate, arxiv.SortCriterion.LastUpdatedDate]:
                print(
                    "For date filters, it is recommended that you sort by submission or updated date.")

            if limit < self.FILTER_BY_DATE_LIMIT:
                limit = self.FILTER_BY_DATE_LIMIT
                print(
                    f"Changing query limit to {self.FILTER_BY_DATE_LIMIT} as filtering by dates require more results.")

            if start_date is None or end_date is None:
                print("Cannot filter by date if start or end date is empty.")
                return
            try:
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            except:
                print("The start and end format should be %Y-%m-%d -> e.g. 2022-01-28.")
                return

        search = arxiv.Search(
            query=search_query,
            max_results=limit,
            sort_by=sort_by
        )

        data_dict = {
            "entry_id": [],
            "updated": [],
            "published": [],
            "title": [],
            "authors": [],
            "summary": [],
            "comment": [],
            "journal_ref": [],
            "doi": [],
            "primary_category": [],
            "categories": [],
            "links": [],
            "pdf_url": []
        }

        for result in search.results():
            if filter_by_date:
                if not self.__is_in_date_range(result.published, start_date, end_date):
                    continue
            for column in data_dict.keys():
                data_dict[column].append(eval(f"result.{column}"))

        data_dict["published_date"] = data_dict.pop("published")
        data_dict["abstract"] = data_dict.pop("summary")

        if save_to_csv:
            pd.DataFrame.from_dict(data_dict).to_csv('arxiv_scraped.csv')

        return data_dict


class ScienceOpen:
    # API-related constants.
    HOST = "https://www.scienceopen.com/"
    SEARCH_PATH = "search"

    # Query constants
    SEARCH_PATH_QUERY = "#('v'~4_'id'~''_'queryType'~2_'context'~null_'kind'~77_'order'~0_'orderLowestFirst'~false_'query'~'"
    FILTERS_AFTER_QUERY = "'_'filters'~!"

    ABSTRACT_FILTER_QUERY = "('kind'~33_'not'~false_'query'~'"
    DATE_FILTER_BEGINNING = "('kind'~37_'not'~false_'dateFrom'~"
    DATE_FILTER_ENDING = "_'dateTo'~"
    END_QUERY_FILTER_PATH = "'_'queryType'~1)"

    CONTENT_TYPE_ARTICLE_FILTER = "('kind'~123_'contentTypes'~!'article'*)"

    END_QUERY_PATH = "*_'hideOthers'~false)"

    MAX_LIMIT = 10

    def __init__(self) -> None:
        pass

    # TODO: get more data? like title, etc
    def get_data_from_query(self, keyword, abstract=None, start_date=None, end_date=None, limit_search_attempt=MAX_LIMIT):
        if len(keyword) <= 0:
            print("Cannot search with empty query.")
            return

        search_query = f"{self.HOST}{self.SEARCH_PATH}{self.SEARCH_PATH_QUERY}{keyword}{self.FILTERS_AFTER_QUERY}"

        if abstract is not None:
            search_query += f"{self.ABSTRACT_FILTER_QUERY}{abstract}{self.END_QUERY_FILTER_PATH}_"

        if start_date is not None or end_date is not None:
            try:
                start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
                end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            except:
                print("The start and end format should be %Y-%m-%d -> e.g. 2022-01-28.")
                return
            start_date_unix = time.mktime(start_date.timetuple()) * 1000
            end_date_unix = time.mktime(end_date.timetuple()) * 1000
            search_query += f"{self.DATE_FILTER_BEGINNING}{start_date_unix}{self.DATE_FILTER_ENDING}{end_date_unix})_"

        search_query += f"{self.CONTENT_TYPE_ARTICLE_FILTER}{self.END_QUERY_PATH}"
        print(search_query)

        attempt_count = 1
        options = Options()
        options.headless = True
        ua = UserAgent()
        user_agent = ua.random
        options.add_argument(f'user-agent={user_agent}')
        driver = webdriver.Chrome(service=Service(
            ChromeDriverManager().install()), options=options)
        driver.get(search_query)
        driver.implicitly_wait(10)

        # Click Allow All Cookies
        element_present = EC.element_to_be_clickable(
            (By.XPATH, "/html/body/aside/div/div/div[2]/div[2]/div/div[2]/div[1]/button[1]"))
        try:
            WebDriverWait(driver, 10).until(element_present)
        except:
            print("Failed to click allow all cookies.")
            driver.close()
            return
        try:
            driver.find_element(
                By.XPATH, "/html/body/aside/div/div/div[2]/div[2]/div/div[2]/div[1]/button[1]").click()
        except:
            print("Failed to click allow all cookies.")
            print(driver.page_source.encode('utf-8'))
            driver.close()
            return

        # Wait for more load button.
        element_present = EC.element_to_be_clickable(
            (By.XPATH, '//*[@id="id1"]/div/div/div/div[2]/div/div[6]/div[2]/div/button'))
        try:
            WebDriverWait(driver, 10).until(element_present)
        except:
            print("Failed to load articles.")
            driver.close()
            return

        # Click load more button.
        while True:
            if attempt_count >= limit_search_attempt:
                break
            element_present = EC.element_to_be_clickable(
                (By.CSS_SELECTOR, '.so-b3.so--tall.so--centered.so--green-2'))
            try:
                WebDriverWait(driver, 10).until(element_present)
            except:
                print("Can't find load more button!")
                break
            try:
                driver.find_element(
                    By.CSS_SELECTOR, '.so-b3.so--tall.so--centered.so--green-2').click()
            except:
                print("No more load more buttons.")
                break
            attempt_count += 1
            time.sleep(1.5)

        abstracts = self.__get_data_from_source(
            driver.page_source.encode('utf-8'))
        driver.close()
        return abstracts

    def __get_abstracts_from_source(self, data):
        soup = BeautifulSoup(data, "html.parser")
        abstractsObj = soup.find_all('div', class_=re.compile(f".*so-d.*"))
        if abstractsObj is None:
            print("Can't find any abstracts.")
            return []
        return [abstract.text for abstract in abstractsObj if len(abstract.text.split()) > 3]

    def __get_data_from_source(self, data):
        soup = BeautifulSoup(data, "html.parser")

        data = {"abstract": [], "published_date": []}
        abstractsObj = soup.find_all('div', class_=re.compile(f".*so-article-list-item-text.*"))
        if abstractsObj is None:
            print("Can't find any abstracts.")
            return data

        for abstractObjs2 in abstractsObj:
            for abstract in abstractObjs2.find_all('div', class_=re.compile(".*so-d.*")):
                abstract = abstract.text
                if len(abstract.split()) <= 3:
                    continue
                data["abstract"].append(abstract)

            for publicationDetails in abstractObjs2.find_all('div', class_=re.compile(".*so-secondary.*")):
                year_candidates = re.findall(
                    r'\(\w+\)', publicationDetails.text)
                if len(year_candidates) <= 0:
                    continue
                for year in year_candidates:
                    try:
                        publication_year = int(
                            ''.join(filter(str.isdigit, year)))
                    except:
                        continue
                data["published_date"].append(datetime.datetime(
                    year=publication_year, month=12, day=31, hour=23, minute=59, second=59))

        return data


def main():
    scienceopen_db = ScienceOpen()
    data = scienceopen_db.get_data_from_query("text recognition")
    print(data)

    # arxiv_db = Arxiv()
    # arxiv_db.get_data_from_query("image recognition", arxiv.SortCriterion.SubmittedDate, filter_by_date=True, start_date="2022-01-01", end_date="2022-08-30", save_to_csv=True)

    # emerald_db = Emerald()
    # abstracts = emerald_db.get_data_from_query('competitive advantage')
    # print(len(abstracts))

    # core_database = CORE()
    # works = core_database.get_works_by_search_query(""""image recognition" AND fieldsOfStudy:"computer science" AND documentType:"research" AND (yearPublished>=2010 AND yearPublished<=2022)""", 50)
    # if works is not None:
    #     abstracts = core_database.get_fulltext_from_works(save_to_csv=True)
    #     print(abstracts)
    pass


if __name__ == "__main__":
    main()
