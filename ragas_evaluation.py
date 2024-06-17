from ragas.integrations.llama_index import evaluate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from ragas.metrics.critique import harmfulness
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.testset.generator import TestsetGenerator
from llama_index.core import SimpleDirectoryReader


embeddings = OllamaEmbedding(model_name="phi3:latest")  # OpenAIEmbedding()

Settings.embed_model = embeddings

documents = SimpleDirectoryReader(
    "./data/EvaluatingLlmSurveyPaperDataset", required_exts=[".pdf"], recursive=True).load_data()

print(documents)

print("Building the vector index...")
vector_index = VectorStoreIndex.from_documents(documents)


print("Building the query engine...")
generator_llm = Ollama(model="phi3:latest")
query_engine = vector_index.as_query_engine(llm=generator_llm)
