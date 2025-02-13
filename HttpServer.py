import os
import io
import numpy as np
import requests
import collections
import time
from datetime import datetime
from datetime import date
import json
import uuid
import logging
from aiohttp import web
from app import start_run
collections.Callable = collections.abc.Callable
from utils import *
def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)
def _bmeso_tag_to_spans(tags, ignore_labels=None):
    
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bmes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]  

serverSubmitDataset =  'http://127.0.0.1:8181/NERtaskSubmit'

model, wtree, lattice_tokenizer,bigram_tokenizer,label_tokenizer,device = start_run()

async def retrieve(request):
    print ('%s 处理数据'%(time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())))
    try :
        data = await request.json()
        taskId = data["taskId"]
        text = data["text"]

        starttime = data["starttime"]
        endtime = data["endtime"]
        
        if len(text) > 0 :
            text = "".join([Q2B(i) for i in text])
            text_set = text.strip().split("。")
            text_set = [string for string in text_set if len(string)>1]
            ner = []
            for string in text_set:
                string = string+"。"
                words = wtree.get_words(string)
                char_idx = []
                token_dict = lattice_tokenizer.token_dict
                for char in string:
                    if char in token_dict.keys():
                        char_idx.append(token_dict[char][0])
                    else:
                        char_idx.append(token_dict["<unk>"][0])
                word_idx = []
                ps = []
                pe = []
                for word in words:
                    ps.append(word[0])
                    pe.append(word[1])
                    if word[2] in token_dict.keys():
                        word_idx.append(token_dict[word[2]][0])
                    else:
                        word_idx.append(token_dict["<unk>"][0])
                        
                bigrams = []
                for i in range(len(string)):
                    if i == 0:
                        bigrams.append("</s>" + string[i])
                    else:
                        bigrams.append(string[i-1] + string[i])
                bigram_idx = []
                bigram_token_dict = bigram_tokenizer.token_dict

                for bigram in bigrams:
                    if bigram in bigram_token_dict.keys():
                        bigram_idx.append(bigram_token_dict[bigram][0])
                    else:
                        bigram_idx.append(bigram_token_dict["<unk>"][0])
                lattice_input = [char_idx + word_idx]
                ps = [[i for i in range(len(char_idx))] + ps]
                pe = [[i for i in range(len(char_idx))] + pe]
                char_len = [len(char_idx)]
                lattice_len = [len(word_idx)]
                bigram_idx = [bigram_idx]
                inp = (torch.tensor(lattice_input).to(device), torch.tensor([0]).to(device)
                                    , torch.tensor(bigram_idx).to(device), torch.tensor(ps).to(device),torch.tensor(pe).to(device)
                                    , torch.tensor([0]).to(device), torch.tensor(char_len).to(device), torch.tensor(lattice_len).to(device) )
                result = model(inp,False)
                pred = result['pred'].cpu().numpy().tolist()[0]
                label_dict = label_tokenizer.idx_dict
                pred_tag = [label_dict[p] for p in pred]
                tags = _bmeso_tag_to_spans(pred_tag)
                
                for tag in tags:
                    item = string[tag[1][0]:tag[1][1]]
                    if tag[0] =="ns" and item not in ner:
                        ner.append(string[tag[1][0]:tag[1][1]])
            headers = {
                "Content-Type": "application/json"
            };
            Named_Entity = {"NER":ner}
            Named_Entity['taskid'] = taskId;
            requests.post(serverSubmitDataset, headers = headers, data = json.dumps(Named_Entity));
        else:
            print(len(text))
    except Exception as e:
        print(e);        
    finally :
        None;
    text = "提交任务"
    return web.Response(text=text)

async def hello(request):
    return web.Response(text="Hello, world")

if __name__ == '__main__':
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:41091'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:41091'
    # textnetServer = "http://127.0.0.1:8181/"
    
    app = web.Application()
    
    app.add_routes([web.get('/retrieve', retrieve),
        web.route('*', '/retrieve', retrieve),
        web.get('/', hello)])

    web.run_app(app, port=8080)