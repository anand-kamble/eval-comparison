
# %%
import json
import time
from typing import Any, List
from llama_index.core.base.base_retriever import BaseRetriever
from numpy import equal
from opik import Dataset, Opik, track, DatasetItem
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from opik.evaluation import evaluate, types
from opik.evaluation.evaluation_result import EvaluationResult
from opik.evaluation.metrics import Equals, Hallucination
from opik.integrations.openai import track_openai
from opik.integrations.llama_index import LlamaIndexCallbackHandler
from llama_index.llms.ollama import Ollama
# Define the task to evaluate
# %%
QUERY_MODEL = "llama3.1"
EVALUATION_MODEL = "llama3.1"
DATASET = "PaulGrahamEssayDataset"

# %%
embeddings = OllamaEmbedding(
    model_name=QUERY_MODEL, base_url="http://localhost:11434")
Settings.embed_model = embeddings
llm = Ollama(model=QUERY_MODEL, request_timeout=600.0,
             base_url="http://localhost:11434", additional_kwargs={"max_length": 512})
# %%
@track
def your_llm_application(input: str) -> str:
    response: CompletionResponse = llm.complete(input)

    return response.text



# %%


documents: List[Document] = SimpleDirectoryReader(
    f"../data/{DATASET}").load_data()
index: VectorStoreIndex = VectorStoreIndex.from_documents(
    documents, show_progress=True)
retriever: BaseRetriever = index.as_retriever()


@track
def your_context_retriever(input: str) -> List:

    return ["..."]#retriever.retrieve(input)


# %%
# Define the evaluation task
def evaluation_task(x: DatasetItem)  -> dict[str, str]:
    
    return {
        "input": x.input['user_question'],
        "output": your_llm_application(x.input['user_question']),
        "context": your_context_retriever(x.input['user_question'])
    }

# %%
# Create a simple dataset
client = Opik()
dataset: Dataset | None = None
try:
    dataset = client.create_dataset(
        name="my-paul-graham-essay-dataset")
    llama_rag_dataset = None
    with open(f"../data/{DATASET}/rag_dataset.json", "r") as f:
        llama_rag_dataset = json.load(f)

    items_to_insert = []
    for item in llama_rag_dataset["examples"]:

        items_to_insert.append(DatasetItem(
            input={"user_question": item["query"]},
            expected_output={"assistant_answer": item["reference_answer"]}
        ))

    dataset.insert(items_to_insert)
except:
    dataset = client.get_dataset(name="my-paul-graham-essay-dataset")
# %%
# Define the metrics
hallucination_metric = Hallucination()
equals_metric = Equals()
# %%
evaluation: EvaluationResult = evaluate(
    task_threads=1,
    experiment_name="My 1st experiment",
    dataset=dataset,
    task=evaluation_task,
    verbose=1,
    scoring_metrics=[equals_metric,hallucination_metric],
    experiment_config={
        "model": QUERY_MODEL
    }
)

# %%
