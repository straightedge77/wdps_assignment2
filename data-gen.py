import os
import json
import spacy

nlp = spacy.blank("en")
nlp.add_pipe('opentapioca')
nlp.add_pipe('sentencizer')
articles = json.load(open('./data/DocRED/doc.json'))
result=[]
for article in articles:
    if article['doc']:
        item = {}
        item['title'] = article['title']
        doc = nlp(article['doc'])
        data = []
        sentid = {}
        for sent_i, sent in enumerate(doc.sents):
            tokens=[]
            for token in sent:
                tokens.append(token.text)
            data.append(tokens)
            sentid[sent.start] = sent_i
        item['sents'] = data
        vertex = []
        flag = [1 for x in range(len(doc.ents))]
        for i in range(len(doc.ents)):
            if (flag[i] != 0):
                ent1 = doc.ents[i]
                entity = []
                for j in range(i, len(doc.ents)):
                    ent2 = doc.ents[j]
                    if ent1.kb_id_ == ent2.kb_id_:
                        mention = {}
                        mention['name'] = ent2.text
                        mention['sent_id'] = sentid[ent2.sent.start]
                        mention['pos'] = [ent2.start - ent2.sent.start, ent2.end - ent2.sent.start]
                        if ent2.label_ == "PERSON":
                            mention['type'] = "PER"
                        elif ent2.label_ == "ORGLOC":
                            mention['type'] = "LOC"
                        else:
                            mention['type'] = ent2.label_
                        entity.append(mention)
                        flag[j] = 0
                vertex.append(entity)
        item['vertexSet'] = vertex
        result.append(item)
json.dump(result, open('./data/DocRED/test.json', "w"))

