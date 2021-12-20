from django.http import HttpResponse
from django.shortcuts import render

import json
import sys
sys.path.append("..")
import pipeline
args, model, tokenizer, nlp = pipeline.load()

def hello(request):

    request.encoding='utf-8'
    if "query" in request.GET and request.GET["query"]:
        message = "Querying " + request.GET["query"]
        query1 = request.GET["query"]
    else:
        message = ""
        query1 = "Netherlands"
    pipeline.utilize(args, model, tokenizer, nlp, query1)
    context = {}
    context['title'] = "Visualization"
    #context['categories'] = [{"name": "PER"}, {"name": "ORG"}, {"name": "LOC"}]
    with open("web.json", 'r') as rf:
        relations = json.loads(rf.readline())
    context['data'] = relations['data']
    context['links'] = relations['links']
    context['categories'] = relations['categories']
    context['message'] = message
    return render(request, 'graph.html', context)

