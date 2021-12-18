import os
import json
from collections import Counter

relations = json.load(open('./checkpoints/result.json'))
entity_info = json.load(open('./data/DocRED/test.json'))
relation_info = json.load(open('./data/DocRED/rel_info.json'))
entity = entity_info[0]['vertexSet']
result = {}
data = []
point = []
links = []
for relation in relations:
    point.append(relation['h_idx'])
    point.append(relation['t_idx'])
    link = {}
    link['source'] = entity[relation['h_idx']][0]['name']
    link['target'] = entity[relation['t_idx']][0]['name']
    link['value'] = ""
    links.append(link)
temp = []
for item in links:
    if not item in temp:
        temp.append(item)
nodes = Counter(point)
for key in nodes:
    node = {}
    node['name'] = entity[key][0]['name']
    node['symbolSize'] = nodes[key]
    if entity[key][0]['type'] == "PER":
        node['category'] = 1
    elif entity[key][0]['type'] == "ORG":
        node['category'] = 2
    else:
        node['category'] = 3
    label = {}
    label['show'] = True
    label['fontSize'] = 20
    node['label'] = label
    data.append(node)
result['data'] = data
for relation in relations:
    source = entity[relation['h_idx']][0]['name']
    target = entity[relation['t_idx']][0]['name']
    for item in temp:
        if source == item['source'] and target == item['target']:
            if item['value'] == "":
                item['value'] = item['value'] + relation_info[relation['r']]
            else:
                item['value'] = item['value'] + " | " + relation_info[relation['r']]
result['links'] = temp
categories = []
category = {}
category['name'] = "PER"
categories.append(category)
category = {}
category['name'] = "ORG"
categories.append(category)
category = {}
category['name'] = "LOC"
categories.append(category)
result['categories'] = categories
json.dump(result, open('web.json', "w"))

