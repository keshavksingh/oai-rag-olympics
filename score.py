import logging
import json
import RAG_BOT_OLYMPICS as rbo
import pandas as pd
import openai
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    global collection
    load_dotenv()
    prepRagContext = rbo.prepareContextData()
    documents,ids = prepRagContext.transform()
    collection = rbo.oaiContextDataEmbedding(documents,ids).OpenAI_ContextData_Embedding()
    logging.info("Init complete")

def run(promptQuery: str):
    print(promptQuery)
    rag = rbo.retrievalAugmentedGeneration(promptQuery)
    result_json = json.dumps({"result":str(rag.getChatResponse(collection))})
    print("Output Response String - !")
    print(result_json)
    logging.info("Request processed")
    return result_json

#if __name__=="__main__":
#    init()
#    print(run("Which individual won the Bronze Medal for Olympics Ice Hockey Mens Ice Hockey in 2002?"))