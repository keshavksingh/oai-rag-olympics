# Import Uvicorn & the necessary modules from FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
# Import other necessary packages
import RAG_BOT_OLYMPICS as rbo
import pandas as pd
import openai
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import json
# Load the environment variables from the .env file into the application
load_dotenv() 
# Initialize the FastAPI application
app = FastAPI()

prepRagContext = rbo.prepareContextData()
documents,ids = prepRagContext.transform()
collection = rbo.oaiContextDataEmbedding(documents,ids).OpenAI_ContextData_Embedding()

# Create the POST endpoint with path '/queryOlympics'
@app.post("/queryOlympics")
async def oaiChatbot(promptQuery: str):
    numberOfResults=5
    rag = rbo.retrievalAugmentedGeneration(promptQuery,numberOfResults)
    result_json = json.dumps({"result":str(rag.getChatResponse(collection))})
    print("Output Response String - !")
    return result_json

@app.post("/searchOlympics")
async def oaiSearch(promptQuery: str):
    numberOfResults=5
    rag = rbo.retrievalAugmentedGeneration(promptQuery,numberOfResults)
    result_json = json.dumps({"result":str(rag.generate_prompt_embedding(collection))})
    print("Output Response String - !")
    return result_json

if __name__ == '__main__':
    app.run(debug=True)