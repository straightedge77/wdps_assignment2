import requests
import json
import sys

# this function is used to crawl wikipedia title and abstract from the Internet
def crawl(query):
    result = []
    doc = {}
    # the offcial api of the wikipedia
    url = 'https://en.wikipedia.org/w/api.php'
    params1 = {
        'action': 'query',
        'format': 'json',
        'list':'search',
        'utf8':1,
        'srsearch':query,
    }
    # First perform a fuzzy search
    data = requests.get(url, params=params1).json()
    # Get the candidate pages and select the first ranked Wikipedia page in the search engine
    subject = data['query']['search'][0]['title']
    doc['title'] = subject
    params = {
        'action': 'query',
        'format': 'json',
        'titles': subject,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
    }
    # Get the information of this page
    response = requests.get(url, params=params)
    data = response.json()
    # Get the title and abstract of this wikipedia page
    page = next(iter(data['query']['pages'].values()))
    doc['doc'] = page['extract'].replace("\n", "")
    result.append(doc)
    return result
