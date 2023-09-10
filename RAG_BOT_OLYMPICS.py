#https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results
import pandas as pd
import openai
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
load_dotenv()

class prepareContextData:
    def __init__(self):
        self.contextDataPath = os.getenv("CONTEXT_DATA_PATH")
    
    def transform(self):
        df=pd.read_csv(self.contextDataPath)
        df=df.loc[df['Year'] == 2002]
        df = df.astype(str)
        df['Gender'] = ["Male" if x =='M' else "Female" for x in df['Sex']]
        df['Medal'] = ["but did not win any medal" if x =='nan' else "won a "+x+" medal" for x in df['Medal']]
        df['Height'] = ["Unknown" if x =='nan' else x for x in df['Height']]
        df['Weight'] = ["Unknown" if x =='nan' else x for x in df['Weight']]

        df['text'] = df['Name'] + ' a ' + df['Gender'] + ', aged ' + df['Age'] + ', height '+df['Height']\
                    +' centimeters, Weight '+df['Weight']+' kilograms, from team '+df['Team']\
                    +' , participated in olympic games held in the year '+df['Year']+' , for the '\
                    +df['Season']+' season, hosted in the city '+ df['City']+' in the '+'"'+df['Sport']+'"'\
                    +' sporting category'\
                    +' for the event '+'"'+df['Event']+'"'+' '+df['Medal']
        df = df.head(100)
        docs=df["text"].tolist() 
        docs= [item.replace('"', '') for item in docs]
        docs= [item.replace("'", '') for item in docs]
        ids= [str(x) for x in df.index.tolist()]
        return (docs,ids)

class oaiContextDataEmbedding:
    
    def __init__(self,documents:list,ids:list):
        self.documents = documents
        self.ids = ids
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def OpenAI_ContextData_Embedding(self):
        openai_embeddingFunction = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name="text-embedding-ada-002")

        client = chromadb.Client()
        collection = client.get_or_create_collection("olympics",embedding_function=openai_embeddingFunction)
        collection.add(documents=self.documents,ids=self.ids)
        return collection


class retrievalAugmentedGeneration:

    def __init__(self,prompt:str,num_results:int):
        #self.documents = documents
        #self.ids = ids
        self.prompt = prompt
        self.num_results=num_results
        #self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def generate_prompt_embedding(self,collection):
        #collection = oaiContextDataEmbedding(self.documents,self.ids).OpenAI_ContextData_Embedding()
        response = openai.Embedding.create(model="text-embedding-ada-002", input=self.prompt)
        results=collection.query(
        query_embeddings= response["data"][0]["embedding"],
        n_results=self.num_results,
        include=["documents"])
        res = "\n".join(str(item) for item in results['documents'][0])
        augmented_prompt=res+" "+self.prompt
        return augmented_prompt
    
    def getChatResponse(self,collection):
        messages = [
        {"role": "system", "content": "You answer questions only about Olympics and nothing else."},
        {"role": "user", "content": self.generate_prompt_embedding(collection)}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        response_message = response["choices"][0]["message"]["content"]
        return response_message

if __name__=="__main__":
    prepRagContext = prepareContextData()
    documents,ids = prepRagContext.transform()
    promptQuery = "Provide the details of the athelete who won a Gold Medal in Olympics 2002?"
    #"Which Actory played Maverick in the movie Top Gun?"
    #"Which individual won the Bronze Medal for Olympics Ice Hockey Mens Ice Hockey in 2002?"
    numberOfResults=5
    collection = oaiContextDataEmbedding(documents,ids).OpenAI_ContextData_Embedding()
    rag = retrievalAugmentedGeneration(promptQuery,numberOfResults)
    print(rag.getChatResponse(collection))
