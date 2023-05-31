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

modelPath = "scr/model"   # Path to the saved model


corpus = read_list() # List of drug names

f = open('scr/databank.json') # Opening our saved JSON file
data_bank = json.load(f) # Loading the JSON object as a dictionary

columns = ['drug_name', 'medical_condition', 'side_effects', 'generic_name',
           'drug_classes', 'brand_names', 'activity', 'rx_otc',
           'pregnancy_category', 'csa', 'alcohol',
           'medical_condition_description', 'rating', 'no_of_reviews']


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/extract_review")
async def predict(prompt: str, focus: list):
    
    # Load the SentenceTransformer model
    model = SentenceTransformer(modelPath)
    
    # Encode all the drug names in the corpus
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True) 
    
    
    query_embedding = model.encode(prompt, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest scores
    top_k = 1
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    top_results = torch.topk(cos_scores, k=top_k)
    
    score, idx = top_results[0], top_results[1]  # Get score and index
    
    drug_name = corpus[idx]  # Retrieve the drug name

    work_data = data_bank[drug_name] # Extract information about the drug

    focus = focus   # Major interest of the prompter

    sub_result = {}

    for i in focus:
        if i in columns:
            sub_result[i] = work_data[i]
            
        else:
            sub_result[i] = 'Not a valid focus'   # Focus not in the column list.
            
            
    for i in columns:
        if i not in focus:
            sub_result[i] = work_data[i]
            
        
    final_result = {"result": 
                [
                    sub_result
                ]
               }

    return final_result
    
    
    
    
    
    
    
    
    


