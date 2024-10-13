import os
from getpass import getpass
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext)

from llama_index.vector_stores.pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# doc = SimpleDirectoryReader('PlotLot/Sec._6.1___Zoning_districts_established..docx').load_data()
# vector_store = PineconeVectorStore(pinecone_index=index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     doc, storage_context=storage_context
# )
# batch size parameter aligns with Pinecone upsertion
embed_model = OpenAIEmbedding()
api_key = '6f3e21bd-05cf-44cb-86d4-4b74de6b8499'
pc = Pinecone(api_key=api_key)

dims = len(embed_model.get_text_embedding("some random text"))


# Create a serverless index
index_name = "rei-index"
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud='aws', 
#             region='us-east-1'
#         ) 
#     ) 
# # connect to index
index = pc.Index(index_name)
time.sleep(1)
# # view index stats
index.describe_index_stats()

doc = SimpleDirectoryReader(input_files=['PlotLot/Sec._6.1___Zoning_districts_established..docx']).load_data()
vector_store = PineconeVectorStore(pinecone_index=index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    doc, storage_context=storage_context
)
# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("""
                              property: 303 s ridge st dallas nc
                              Width: 80
                              Length: 200
                              Zoning: I2
                              Based on these property descriptions tell me what is the maximum allowable units for this lot
                              """)
print(response)