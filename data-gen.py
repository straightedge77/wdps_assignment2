import os
import json
import spacy

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker", last=True)
articles = json.load(open('./data/DocRED/doc.json'))
result=[]
for article in articles:
    if article['doc']:
        item = {}
        item['title'] = article['title']
        doc = nlp(article['doc'])
        all_linked_entities = doc._.linkedEntities
        entities = []
        for linked_entity in all_linked_entities:
            entity1 = linked_entity.get_span()
            for entity2 in doc.ents:
                if entity1.start == entity2.start:
                    enti = {}
                    enti['entity'] = linked_entity
                    enti['ent'] = entity2
                    entities.append(enti)
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
        flag = [1 for x in range(len(entities))]
        for i in range(len(entities)):
            if (flag[i] != 0):
                ent1 = entities[i]['entity']
                entity = []
                for j in range(i, len(entities)):
                    ent2 = entities[j]['entity']
                    if ent1.get_id() == ent2.get_id():
                        ent = entities[j]['ent']
                        mention = {}
                        mention['name'] = ent.text
                        mention['sent_id'] = sentid[ent.sent.start]
                        mention['pos'] = [ent.start - ent.sent.start, ent.end - ent.sent.start]
                        if ent.label_ == "GPE":
                            mention['type'] = "LOC"
                        elif ent.label_ == "PERSON":
                            mention['type'] = "PER"
                        elif ent.label_ == "ORG":
                            mention['type'] = "ORG"
                        elif ent.label_ == "NUM":
                            mention['type'] = "NUM"
                        elif ent.label_ == "TIME":
                            mention['type'] = "TIME"
                        elif ent.label_ == "PAD":
                            mention['type'] = "PAD"
                        else:
                            mention['type'] = "MISC"
                        entity.append(mention)
                        flag[j] = 0
                vertex.append(entity)
        item['vertexSet'] = vertex
        result.append(item)
json.dump(result, open('./data/DocRED/test.json', "w"))

