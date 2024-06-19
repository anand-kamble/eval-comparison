import time
from ragas.integrations.llama_index import evaluate
from ragas.metrics.critique import harmfulness
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,

)
import json
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from llama_index.core import SimpleDirectoryReader
from datasets import Dataset

time_dict = {}
start_time = time.time()

# embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # OpenAIEmbedding()
embeddings = OllamaEmbedding(model_name="phi3:latest")
Settings.embed_model = embeddings
end_time = time.time()
time_dict['embedding_setup'] = end_time - start_time
print("Time taken for embedding setup: ", time_dict['embedding_setup'])

start_time = time.time()
documents = SimpleDirectoryReader(
    "./data/EvaluatingLlmSurveyPaperDataset", required_exts=[".pdf"], recursive=True).load_data()
end_time = time.time()
time_dict['document_loading'] = end_time - start_time
print("Time taken for document loading: ", time_dict['document_loading'])

print("Building the vector index...")
start_time = time.time()
vector_index = VectorStoreIndex.from_documents(documents[:2])
end_time = time.time()
time_dict['vector_index_building'] = end_time - start_time
print("Time taken for vector index building: ", time_dict['vector_index_building'])

print("Building the query engine...")
start_time = time.time()
generator_llm = Ollama(model="phi3:latest",request_timeout=600.0)
query_engine = vector_index.as_query_engine(llm=generator_llm)
end_time = time.time()
time_dict['query_engine_building'] = end_time - start_time
print("Time taken for query engine building: ", time_dict['query_engine_building'])

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    harmfulness,
    # max_workers = 24 # Found this argument from the RunConfig dataclass.
]

critic_llm = Ollama(model="llama3",request_timeout=600.0)  # OpenAI(model="gpt-4")
# using GPT 3.5, use GPT 4 / 4-turbo for better accuracy
evaluator_llm = critic_llm  # OpenAI(model="gpt-3.5-turbo")
# USING CRITIC LLM TO KEEP EVERYTHING LOCAL FOR NOW.


# For testset needs to be a dictionary with keys "question" and "ground_truth"
# From which the question and ground_truth are lists of strings.
# testset = {
#     "question": [
#         "What are the types of clusters?",
#         "What is Fuzzy clustering?",
#         "What is the difference between K-means and Hierarchical clustering?",
#     ],
#     "ground_truth": [
#         "The types of clusters mentioned are: well-separated clusters, prototype-based clusters, contiguity-based clusters, density-based clusters, and clusters defined by an objective function.",
#         "In fuzzy clustering, a data point can belong to multiple clusters with weights between 0 and 1, where the weights for each point sum up to 1. It's a type of non-exclusive clustering.",
#         "K-means clustering is a partitional clustering approach that divides the data into non-overlapping subsets (clusters), while hierarchical clustering produces a set of nested clusters organized as a hierarchical tree. K-means requires specifying the number of clusters in advance, while hierarchical clustering does not. K-means typically has a global objective function, while hierarchical clustering algorithms have local objectives.",
#     ]}
# To make this testset from the llama Dataset, we can convert the dataset to a dictionary as follows:
start_time = time.time()

llama_rag_dataset = None
with open("data/EvaluatingLlmSurveyPaperDataset/rag_dataset.json", "r") as f:
    llama_rag_dataset = json.load(f)

testset = {
    "question": [],
    "ground_truth": [],
}


# Here we are using the reference answer as the ground truth.
for item in llama_rag_dataset["examples"][:5]:
    testset["question"].append(item["query"])
    testset["ground_truth"].append(item["reference_answer"])


    """
    Here I realized that for evluation we need to provide a Dataset object to the ragas evaluate function.
    Not just the dictionary. So I will convert the dictionary to a Dataset object.
    
    I found this by look at the source code, and the type given 
    """

dataset = Dataset.from_dict(testset)

end_time = time.time()
time_dict['testset_loading'] = end_time - start_time
print("Time taken for testset loading: ", time_dict['testset_loading'])


print("Evaluating the query engine...")
start_time = time.time()
result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=testset,
    llm=evaluator_llm,
    embeddings=OllamaEmbedding(model_name="phi3:latest"),
)

end_time = time.time()
time_dict['evaluation'] = end_time - start_time
print("Time taken  for evaluation: ", time_dict['evaluation'])

result.to_pandas().to_csv("ragas_evaluation.csv")


# Save timing results to a text file
with open("timing_results.txt", "w") as f:
    for key, value in time_dict.items():
        f.write(f"{key}: {value} seconds\n")
