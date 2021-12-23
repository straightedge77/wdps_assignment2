# wdps_assignment2 - Knowledge Graph Construction

## Code Introduction
- web.py is used to crawl title and abstract from Wikipedia
- data-gen.py is used to convert the raw data into data that can be used by the model，the result is stored at ./data/DocRED/test.json
- train.sh is used to train the model，the model trained is stored at ./checkpoints
- app.py is used to convert the result of the model to data that can be used in Visualization
- pipeline.py is used to load the model and make predication
- manage.py is used to start the Web Server

## Prerequisites

```
# Installing the Prerequisites
python==3.8(recommened create virtual environment using conda)
pytorch==1.8.2(recommend installed using conda and following the instruction of https://pytorch.org/get-started/locally/ )
cuda==11.1
transformers==2.7.0(recommmend installed using pip)
django
spacy==3.2.0(recommend installed using pip)
spacy-entity-linker(recommend installed using pip)
(Download the model and dataset for entity linking)
python -m spacy download en_core_web_md
python -m spacy_entity_linker "download_knowledge_base"
 ```
Download pretrained model from these two links:
https://drive.google.com/file/d/1Z_aR1BhJSYZCkW6rn5mWPjkAz3y2LEQ4/view?usp=sharing
https://drive.google.com/file/d/1eBRHffGIWzxnpHyKjZvno4DHGj1l0xUq/view?usp=sharing
unzip these two files and put them in directory where contains the code

## Run
```
python manage.py (open 127.0.0.1:8000 in browser)
```
If the program return errors like the thread is occupied, just restart the program and it will all ok.
