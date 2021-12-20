# wdps_assignment2 - Knowledge Graph Construction

## Code Introduction
- web.py是用于从wikipedia上爬取词条的title以及abstract,文件存储在./data/DocRED/doc.json
- data-gen.py是用于将爬取的信息转换为模型可以使用的数据，文件存储在./data/DocRED/test.json
- predict.sh是用于使用模型来进行关系提取，文件存储在./checkpoints/result.json
- app.py是用于将结果生成可以运行在网络上的echarts
- manage.py用于生成网页服务器

## Prerequisites
DAS cluster上有创建好的conda虚拟环境assignment2.1, 代码在/var/scratch/wdps2106/SSAN/wdps2/code里面

```
# 自己配置环境(用pip或者conda装的部分)
python==3.8
pytorch==1.8.2
cuda==11.1
transformers==2.7.0
django
spacy==3.2.0
spacy-entity-linker
(下载模型和数据库)
python -m spacy download en_core_web_md
python -m spacy_entity_linker "download_knowledge_base"
 ```

## Run
如果在DAS cluster中
```
srun --time=01:30:00 -C TitanX --gres=gpu:1 --pty /bin/bash
conda activate assignment2.1
module load cuda11.1/toolkit
python manage.py 然后打开127.0.0.1:8000即可
```
如果在自己的电脑上
```
python manage.py 然后打开127.0.0.1:8000即可
```


如果自己创建环境,首先通过这两个链接https://drive.google.com/file/d/1Z_aR1BhJSYZCkW6rn5mWPjkAz3y2LEQ4/view?usp=sharing,https://drive.google.com/file/d/1eBRHffGIWzxnpHyKjZvno4DHGj1l0xUq/view?usp=sharing下载文件
```
unzip两个文件，并且将其放到代码的子目录下
sh run.sh /subject/you/want/to/search
```