import argparse
import glob
import json
import logging
import os
import random
import requests
import sys
import os
from collections import Counter
import spacy

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from dataset import docred_convert_examples_to_features as convert_examples_to_features
from dataset import DocREDProcessor

from model import (BertForDocRED, RobertaForDocRED)
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker", last=True)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def predict(args, model, tokenizer, prefix=""):
    processor = DocREDProcessor()
    pred_examples = processor.get_test_examples(args.data_dir)

    label_map = processor.get_label_map(args.data_dir)
    predicate_map = {}
    for predicate in label_map.keys():
        predicate_map[label_map[predicate]] = predicate

    eval_dataset = load_and_cache_examples(args, tokenizer, predict=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    ent_masks = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type == 'bert' else None,
                      "ent_mask": batch[3],
                      "ent_ner": batch[4],
                      "ent_pos": batch[5],
                      "ent_distance": batch[6],
                      "structure_mask": batch[7],
                      "label": batch[8],
                      "label_mask": batch[9],
                      }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            ent_masks = inputs["ent_mask"].detach().cpu().numpy()
            out_label_ids = inputs["label"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            ent_masks = np.append(ent_masks, inputs["ent_mask"].detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["label"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    print("eval_loss: {}".format(eval_loss))
    output_preds = []
    for (i, (example, pred, ent_mask)) in enumerate(zip(pred_examples, preds, ent_masks)):
        for h in range(len(example.vertexSet)):
            for t in range(len(example.vertexSet)):
                if h == t:
                    continue
                if np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                    continue
                for predicate_id, logit in enumerate(pred[h][t]):
                    if predicate_id == 0:
                        continue
                    else:
                        output_preds.append((logit, example.title, h, t, predicate_map[predicate_id]))
    output_preds.sort(key=lambda x: x[0], reverse=True)
    output_preds_thresh = []
    for i in range(len(output_preds)):
        if output_preds[i][0] < args.predict_thresh:
            break
        output_preds_thresh.append({"title": output_preds[i][1],
                                    "h_idx": output_preds[i][2],
                                    "t_idx": output_preds[i][3],
                                    "r": output_preds[i][4],
                                    "evidence": []
                                    })
    # write pred file
    if not os.path.exists('./data/DocRED/') and args.local_rank in [-1, 0]:
        os.makedirs('./data/DocRED')
    output_eval_file = os.path.join(args.checkpoint_dir, "result.json")
    with open(output_eval_file, 'w') as f:
        json.dump(output_preds_thresh, f)


def load_and_cache_examples(args, tokenizer, evaluate=False, predict=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = DocREDProcessor()
    # Load data
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_map = processor.get_label_map(args.data_dir)

    if evaluate:
        examples = processor.get_dev_examples(args.data_dir)
    elif predict:
        examples = processor.get_test_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    features = convert_examples_to_features(
        examples,
        args.model_type,
        tokenizer,
        max_length=args.max_seq_length,
        max_ent_cnt=args.max_ent_cnt,
        label_map=label_map
    )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_ent_mask = torch.tensor([f.ent_mask for f in features], dtype=torch.float)
    all_ent_ner = torch.tensor([f.ent_ner for f in features], dtype=torch.long)
    all_ent_pos = torch.tensor([f.ent_pos for f in features], dtype=torch.long)
    all_ent_distance = torch.tensor([f.ent_distance for f in features], dtype=torch.long)
    all_structure_mask = torch.tensor([f.structure_mask for f in features], dtype=torch.bool)
    all_label = torch.tensor([f.label for f in features], dtype=torch.bool)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.bool)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_ent_mask, all_ent_ner, all_ent_pos, all_ent_distance,
                            all_structure_mask, all_label, all_label_mask)

    return dataset

def crawl(query):
    result = []
    doc = {}
    url = 'https://en.wikipedia.org/w/api.php'
    params1 = {
        'action': 'query',
        'format': 'json',
        'list':'search',
        'utf8':1,
        'srsearch':query,
    }
 
    data = requests.get(url, params=params1).json()
    subject = data['query']['search'][0]['title']
    doc['title'] = subject
    params = {
        'action': 'query',
        'format': 'json',
        'titles': subject,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
    }

    response = requests.get(url, params=params)
    data = response.json()
 
    page = next(iter(data['query']['pages'].values()))
    doc['doc'] = page['extract']
    result.append(doc)
    return result

def generate(articles):
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

def web():
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
    json.dump(result, open('./Visualization/web.json', "w"))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_ent_cnt",
        default=42,
        type=int,
        help="The maximum entities considered.",
    )
    parser.add_argument("--no_naive_feature", action="store_true",
                        help="do not exploit naive features for DocRED, include ner tag, entity id, and entity pair distance")
    parser.add_argument("--entity_structure", default='biaffine', type=str, choices=['none', 'decomp', 'biaffine'],
                        help="whether and how do we incorporate entity structure in Transformer models.")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run pred on the pred set.")
    parser.add_argument("--predict_thresh", default=0.5, type=float, help="pred thresh")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=30, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup ratio, overwriting warmup_steps.")
    parser.add_argument("--lr_schedule", default='linear', type=str, choices=['linear', 'constant'],
                        help="Linear warmup ratio, overwriting warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    ModelArch = None
    if args.model_type == 'roberta':
        ModelArch = RobertaForDocRED
    elif args.model_type == 'bert':
        ModelArch = BertForDocRED

    if args.no_naive_feature:
        with_naive_feature = False
    else:
        with_naive_feature = True

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    processor = DocREDProcessor()
    label_map = processor.get_label_map(args.data_dir)
    num_labels = len(label_map.keys())

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

 # predict
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, do_lower_case=args.do_lower_case)
        model = ModelArch.from_pretrained(args.checkpoint_dir,
                                          from_tf=bool(".ckpt" in args.model_name_or_path),
                                          config=config,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          num_labels=num_labels,
                                          max_ent_cnt=args.max_ent_cnt,
                                          with_naive_feature=with_naive_feature,
                                          entity_structure=args.entity_structure,
                                          )
        model.to(args.device)
        query1 = "France"
        print("Downloading document...")
        articles1 = crawl(query1)
        print("Converting into data...")
        generate(articles1)
        print("Extracting relationship...")
        predict(args, model, tokenizer)
        print("Converting into web data...")
        web()
        query2 = "Italy"
        print("Downloading document...")
        articles2 = crawl(query2)
        print("Converting into data...")
        generate(articles2)
        print("Extracting relationship...")
        predict(args, model, tokenizer)
        print("Converting into web data...")
        web()

if __name__ == "__main__":
    main()
