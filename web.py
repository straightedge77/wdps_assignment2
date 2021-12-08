import requests
import json

result = []
doc = {}
query = 'Milan'
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
doc['doc'] = page['extract']
result.append(doc)
json.dump(result, open('doc3.json', "w"))

