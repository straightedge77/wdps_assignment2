import os
import json

# this function is to perform NLP Preprocessing, NER and entity linking to generate the desired data for the model
def generate(articles, nlp):
    result=[]
    for article in articles:
        if article['doc']:
            item = {}
            item['title'] = article['title']
            # Use the spacy pipeline to perform NER and entity linking, more information in pipeline.py
            doc = nlp(article['doc'])
            all_linked_entities = doc._.linkedEntities
            entities = []
            # Because the entity linking pipeline doesn't record down the label of the entity and generate too much entities
            for linked_entity in all_linked_entities:
                entity1 = linked_entity.get_span()
                for entity2 in doc.ents:
                    # We only select the entities that are recognized by the en_core_web_md and this entityLinker pipeline
                    if entity1.start == entity2.start:
                        enti = {}
                        enti['entity'] = linked_entity
                        enti['ent'] = entity2
                        entities.append(enti)
            data = []
            sentid = {}
            # extract the tokens based on the sentences
            for sent_i, sent in enumerate(doc.sents):
                tokens=[]
                for token in sent:
                    tokens.append(token.text)
                data.append(tokens)
                sentid[sent.start] = sent_i
            item['sents'] = data
            vertex = []
            # record down the entity information
            flag = [1 for x in range(len(entities))]
            for i in range(len(entities)):
                if (flag[i] != 0):
                    ent1 = entities[i]['entity']
                    entity = []
                    # try to find there are mentions linked to the same item in the Wikidata, if so we group them in the list, this can imporve the performance of the model
                    for j in range(i, len(entities)):
                        ent2 = entities[j]['entity']
                        if ent1.get_id() == ent2.get_id():
                            ent = entities[j]['ent']
                            mention = {}
                            # record down entity's name, category, description, which sentence it is in and its position in the sentence, this are all required information in the model
                            mention['name'] = ent.text
                            mention['sent_id'] = sentid[ent.sent.start]
                            mention['description'] = ent2.get_description()
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
