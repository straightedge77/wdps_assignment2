import os
import json
from collections import Counter

# this function is to transform the result of the model into the data that can be visualized
def web():
    # the result of the model only contains the id of the source, target and relation, need to transform them into human-readable information
    relations = json.load(open('./checkpoints/result.json'))
    entity_info = json.load(open('./data/DocRED/test.json'))
    relation_info = json.load(open('./data/DocRED/rel_info.json'))
    entity = entity_info[0]['vertexSet']
    result = {}
    data = []
    point = []
    links = []
    # record down the relation and nodes
    for relation in relations:
        point.append(relation['h_idx'])
        point.append(relation['t_idx'])
        link = {}
        link['source'] = entity[relation['h_idx']][0]['name']
        link['target'] = entity[relation['t_idx']][0]['name']
        link['value'] = ""
        links.append(link)
    # there may be multiple relation between two nodes, need to remove the duplicate edges
    temp = []
    for item in links:
        if not item in temp:
            temp.append(item)
    # count the number of the time the nodes appear in the edges
    nodes = Counter(point)
    id={}
    i=0
    # record down the information of the nodes
    for key in nodes:
        node = {}
        node['name'] = entity[key][0]['name']
        # assign an unique id for each node
        node['id'] = i
        # the value of the node is the description of the entity from wikidata
        node['value'] = entity[key][0]['description']
        id[entity[key][0]['name']] = i
        # the size of the model is the number of the time it appear in the relations
        node['symbolSize'] = nodes[key]
        if entity[key][0]['type'] == "PAD":
            node['category'] = 1
        elif entity[key][0]['type'] == "ORG":
            node['category'] = 2
        elif entity[key][0]['type'] == "LOC":
            node['category'] = 3
        elif entity[key][0]['type'] == "NUM":
            node['category'] = 4
        elif entity[key][0]['type'] == "TIME":
            node['category'] = 5
        else:
            node['category'] = 6
        label = {}
        label['show'] = "true"
        label['fontSize'] = 20
        node['label'] = label
        data.append(node)
        i=i+1
    result['data'] = data
    # the value of the edges is the relation information
    for relation in relations:
        source = entity[relation['h_idx']][0]['name']
        target = entity[relation['t_idx']][0]['name']
        for item in temp:
            # if there is multiple relation in one edge, the value should record down all relation information
            if source == item['source'] and target == item['target']:
                if item['value'] == "":
                    item['value'] = item['value'] + relation_info[relation['r']]
                else:
                    item['value'] = item['value'] + " | " + relation_info[relation['r']]
    for item in temp:
        # assign id for source and target
        item['source'] = id[item['source']]
        item['target'] = id[item['target']]
    result['links'] = temp
    # record the information of the categories
    categories = []
    category = {}
    category['name'] = "PAD"
    categories.append(category)
    category = {}
    category['name'] = "ORG"
    categories.append(category)
    category = {}
    category['name'] = "LOC"
    categories.append(category)
    category = {}
    category['name'] = "NUM"
    categories.append(category)
    category = {}
    category['name'] = "TIME"
    categories.append(category)
    category = {}
    category['name'] = "MISC"
    categories.append(category)
    category = {}
    category['name'] = "PER"
    categories.append(category)
    result['categories'] = categories
    json.dump(result, open('web.json', "w"))
