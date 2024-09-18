# %%
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from typing import List
from llama_index.core import Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import AsyncStreamingResponse, PydanticResponse, Response, StreamingResponse
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import Document
from opik.integrations.llama_index import LlamaIndexCallbackHandler
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import opik

# %%
"""
Configure the Opik workspace and API key.

- The `opik.configure()` function sets up the API key and workspace for authentication.
- In this example, the API key and workspace are provided for a user identified as 'anand-kamble'.
"""
opik.configure(api_key="cSXOonbmGcHatvz5631dJGGk1", workspace="anand-kamble")

# %%
"""
Set global constants for the query and evaluation models, and specify the dataset.

- `QUERY_MODEL` and `EVALUATION_MODEL`: Define the Llama model versions to be used for querying and evaluation.
- `DATASET`: Refers to the dataset that contains documents (in this case, "PaulGrahamEssayDataset").
"""
QUERY_MODEL = "llama3.1"
EVALUATION_MODEL = "llama3.1"
DATASET = "PaulGrahamEssayDataset"

# %%
"""
Configure the callback handler and embedding model for the index.

- `opik_callback_handler`: This handler is created to track and manage the query events during runtime.
- `CallbackManager`: A global callback manager is initialized to handle callback operations using `opik_callback_handler`.
- `embeddings`: Ollama-based embedding is created using the specified query model (`QUERY_MODEL`), and an API endpoint is set to interact with the Ollama server at `localhost:11434`.
- These settings are applied to the `Settings` object globally to configure the embedding model and callback manager for the query engine.
"""
opik_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([opik_callback_handler])
embeddings = OllamaEmbedding(
    model_name=QUERY_MODEL, base_url="http://localhost:11434")
Settings.embed_model = embeddings


# %%
"""
This section loads the dataset documents into the index.

- It reads the documents from the specified directory using `SimpleDirectoryReader`.
- The `VectorStoreIndex` is created from the loaded documents.
"""
documents: List[Document] = SimpleDirectoryReader(
    f"../data/{DATASET}").load_data()
index: VectorStoreIndex = VectorStoreIndex.from_documents(documents)


# %%
"""
This section configures the query engine with the specified LLM (Ollama).
- `generator_llm` is the instance of Ollama LLM, configured with `QUERY_MODEL`.
- A query engine is created from the index with the specified LLM as the query engine's generator.
"""
generator_llm = Ollama(model=QUERY_MODEL, request_timeout=600.0,
                       base_url="http://localhost:11434",
                       additional_kwargs={"max_length": 512})
query_engine: BaseQueryEngine = index.as_query_engine(llm=generator_llm)

# %%
"""
This section performs a query using the query engine.

- The query "Who is Paul Graham?" is sent to the query engine.
- The response is captured in one of the possible response types: `Response`, `StreamingResponse`, `AsyncStreamingResponse`, or `PydanticResponse`.
- The response is printed.
"""
response: Response | StreamingResponse | AsyncStreamingResponse | PydanticResponse = query_engine.query(
    "Who is Paul Graham?")
print(response)

# %%
