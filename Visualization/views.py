from django.http import HttpResponse
from django.shortcuts import render

import json
import sys
sys.path.append("..")
import pipeline
# preload the model when the server is up
args, model, tokenizer, nlp = pipeline.load()

def hello(request):

    request.encoding='utf-8'
    # find if perform query
    if "query" in request.GET and request.GET["query"]:
        # if so the things to go through the pipeline is the what we get from query
        query1 = request.GET["query"]
    else:
        # otherwise we will query Netherlands
        query1 = "Netherlands"
    message = query1 + " Relation Graph"
    # conduct the pipeline
    pipeline.utilize(args, model, tokenizer, nlp, query1)
    context = {}
    context['title'] = "Wikipedia abstract Relation Graph Generation"
    #context['categories'] = [{"name": "PER"}, {"name": "ORG"}, {"name": "LOC"}]
    with open("web.json", 'r') as rf:
        relations = json.loads(rf.readline())
    # return information to the server
    context['data'] = relations['data']
    context['links'] = relations['links']
    context['categories'] = relations['categories']
    context['message'] = message
    return render(request, 'graph.html', context)

