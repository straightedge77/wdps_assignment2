from django.http import HttpResponse
from django.shortcuts import render

import json

def hello(request):
    context = {}
    context['title'] = "Visualization"
    #context['categories'] = [{"name": "PER"}, {"name": "ORG"}, {"name": "LOC"}]
    with open("web.json", 'r') as rf:
        relations = json.loads(rf.readline())
    context['data'] = relations['data']
    context['links'] = relations['links']
    context['categories'] = relations['categories']
    return render(request, 'graph.html', context)