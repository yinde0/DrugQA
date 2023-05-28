import random
import torch
from sentence_transformers import SentenceTransformer, util
import fastapi, uvicorn
import io
import requests
from typing import List
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from utils import read_list
import json





app = FastAPI()

modelPath = "scr/model"


corpus = read_list()
# Opening JSON file
f = open('scr/databank.json')
# returns JSON object as a dictionary
data_bank = json.load(f)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/extract_review")
async def predict(prompt: str, focus: list):
    
    
    #print('prompt is ',prompt)
    #print('focus is ',focus)
    
    


    model = SentenceTransformer(modelPath)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)  #encoding all the drug names

    model.save(modelPath)
    model = SentenceTransformer(modelPath)
    
    query_embedding = model.encode(prompt, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest scores
    top_k = 1
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    
    score, idx = top_results[0], top_results[1]
    
    
    drug_name = corpus[idx]

    work_data = data_bank[drug_name]

    focus = focus

    sub_result = {}

    for i in focus:
        result = work_data[i]
        sub_result[i] = result
        
    final_result = {"result": 
                [
                    sub_result
                ]
               }

    return final_result
    
    
    
    
    
    
    
    
    


