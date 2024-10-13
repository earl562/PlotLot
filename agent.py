import os
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext)
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import AgentRunner, ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.pinecone import PineconeVectorStore
from dotenv import load_dotenv
from tools import (
    calculate_max_allowable_units,
    extract_number,
    streamline_variance_application

)
from IPython.display import display, Markdown, Latex

from toolhouse_llamaindex import ToolhouseLlamaIndex
from toolhouse import Toolhouse

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

llm = OpenAI(model="gpt-4o")
embed_model = OpenAIEmbedding()
api_key = PINECONE_API_KEY
pc = Pinecone(api_key=api_key)

dims = len(embed_model.get_text_embedding("some random text"))


# Create a serverless index
index_name = "rei-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 
# # connect to index
index = pc.Index(index_name)
time.sleep(1)
# # view index stats
index.describe_index_stats()

th = Toolhouse()
th.set_metadata("id", "daniele")
th.set_metadata("timezone", -8)
# th.bundle = "search and scrape" # optional, only if you want to use bundles

ToolhouseSpec = ToolhouseLlamaIndex(th)
tool_spec = ToolhouseSpec()

doc = SimpleDirectoryReader(input_files=['Sec._6.1___Zoning_districts_established.docx']).load_data()
vector_store = PineconeVectorStore(pinecone_index=index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    doc, storage_context=storage_context
)
cmau = FunctionTool.from_defaults(fn=calculate_max_allowable_units)
en = FunctionTool.from_defaults(fn=extract_number)
slva = FunctionTool.from_defaults(fn=streamline_variance_application)
tools = [cmau,en,slva]
tools.extend(tool_spec.to_tool_list())
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)

# query_engine = index.as_query_engine()
# response = query_engine.query("""
#                               property: 303 s ridge st dallas nc
#                               Width: 80
#                               Length: 200
#                               Zoning: I2
#                               Based on these property descriptions tell me what is the maximum allowable units for this lot
#                               """)
# print(response)

response = agent.chat('''
Property Details:
Address: 303 S Ridge St, Dallas, NC
County: Gaston County
Lot Dimensions: Width = 80 feet, Length = 200 feet
Zoning: R-8 Residential
Purchase Price: 50,000
Instructions:
Calculate the Maximum Number of Residential Units:

If specific zoning parameters are provided (e.g., density limits, minimum lot area per unit, setback requirements):

Use these parameters to calculate the maximum number of residential units that can be built on the lot according to the R-8 zoning regulations of Gaston County.
If specific zoning parameters are not provided:

Search for and retrieve the missing zoning parameters from reliable sources (e.g., Gaston County zoning ordinances, official county websites).
Use the retrieved information to estimate the maximum number of units that can be built.
Consider all relevant zoning restrictions, including:

Density limits (e.g., maximum units per acre)
Minimum lot area per unit
Setback requirements
Lot width and depth requirements
Any other pertinent zoning ordinances
Assess if It's a Good Deal:

Calculate the Minimum Gross Sale Value using the formula:

Minimum Gross Sale Value = Purchase Price / 0.20 (representing 20% equity)
Determine the average sales price of properties in the area of the specified address:

Search for and retrieve the average sales price data from reliable sources (e.g., real estate databases, recent sales records, market analysis reports).
Compare the Minimum Gross Sale Value to the average area sales price.

Conclude whether the investment is a good deal:

If the Minimum Gross Sale Value is less than or equal to the average area sales price, Good Deal = Yes
Otherwise, Good Deal = No
Output Format:

Number of Units: [Number or 'none']

Reasoning:

Calculations and reasoning for the number of units, including any zoning parameters found and how they were applied.
Sources of the zoning information retrieved.
Good Deal Assessment: [Yes or No]

Explanation:

Calculations and reasoning behind the good deal assessment.
Sources of the average sales price data retrieved.
Example Output:
Number of Units: 2

Reasoning:

Searched and found that R-8 zoning in Gaston County allows for a maximum density of 5 units per acre.
The lot size is 16,000 sq ft (80 ft x 200 ft), which is approximately 0.367 acres.
Maximum units = 0.367 acres x 5 units/acre ≈ 1.83 units, rounded down to 1 unit.
Considering setback requirements and minimum lot size per unit, the lot can accommodate 1 unit.
Sources: Gaston County Zoning Ordinance Section X.X.
Good Deal Assessment: Yes

Explanation:

Minimum Gross Sale Value = $50,000 (Purchase Price) / 0.20 = $250,000.
Average Sales Price in the area is $300,000, retrieved from recent sales data.
Since $250,000 ≤ $300,000, this is considered a good deal.
Sources: Real Estate Database XYZ, accessed on [Date].''')
print(response)