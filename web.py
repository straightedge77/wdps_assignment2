import requests
import json
import sys

result = []
doc = {}
query = "United Kingdom"
url = 'https://en.wikipedia.org/w/api.php'
params1 = {
        'action': 'query',
        'format': 'json',
        'list':'search',
        'utf8':1,
        'srsearch':query,
    }
 
data = requests.get(url, params=params1).json()
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

response = requests.get(url, params=params)
data = response.json()
 
page = next(iter(data['query']['pages'].values()))
doc['doc'] = page['extract'].replace("\n", "")
result.append(doc)
json.dump(result, open('./data/DocRED/doc.json', "w"))

