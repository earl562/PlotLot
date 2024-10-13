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
You are an assistant that provides real estate development analyses.

**Property Details:**

- **Address**: 303 S Ridge St, Dallas, NC
- **County**: Gaston County
- **Lot Dimensions**: Width = 80 feet, Length = 200 feet
- **Zoning**: R-8 Residential
- **Purchase Price**: $50,000

**Instructions:**

1. **Calculate the Maximum Number of Residential Units:**

   - Since specific zoning parameters are not provided, search for and retrieve the missing zoning parameters from reliable sources.
   - Use these parameters to estimate the maximum number of residential units that can be built.
   - Consider all relevant zoning restrictions.

2. **Assess if It's a Good Deal:**

   - Calculate the Minimum Gross Sale Value: $50,000 / 0.20 = $250,000.
   - Search for and retrieve the average sales price of properties in the area.
   - Compare the Minimum Gross Sale Value to the average area sales price.
   - Conclude whether the investment is a good deal.

3. **Output Format:**

   - Provide the final answer **only** in JSON format as specified below, without any additional text.

```json
{
  "number_of_units": [Number or "none"],
  "reasoning": "[Your reasoning here]",
  "good_deal_assessment": "[Yes or No]",
  "explanation": "[Your explanation here]"
}
### **Final Notes:**

- **Ethical Considerations**: Ensure that any data retrieved complies with data use policies and privacy laws.

- **User Inputs**: Validate and sanitize all user inputs to prevent injection attacks or other security issues.

- **Agent Capabilities**: If the agent cannot access real-time data, adjust your approach to provide as much information as possible within the prompt.''')
print(response)