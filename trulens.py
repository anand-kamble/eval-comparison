# %%
# Uncomment the following lines to install the necessary packages
# !pip install trulens_eval llama_index openai
# !pip install "litellm>=1.25.2"

# Import necessary libraries
import numpy as np
import pandas as pd
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.app import App
from trulens_eval.feedback.provider import LiteLLM

# %%
# Initialize the embedding model
embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embeddings

# Load documents and create a vector store index
documents = SimpleDirectoryReader("data/HistoryOfAlexnetDataset").load_data()
index = VectorStoreIndex.from_documents(documents)

# Initialize the LLM and create a query engine
generator_llm = Ollama(model="phi3:latest")
query_engine = index.as_query_engine(llm=generator_llm)

# %%
# Configure the LiteLLM provider for feedback functions
provider = LiteLLM(
    model_engine="ollama/phi3:latest",
    endpoint="http://localhost:11434",
    kwargs={"set_verbose": True},  # Verbose mode for easier debugging
)

# Select the context for the application
context = App.select_context(query_engine)

# %%
# Define feedback functions

# Groundedness feedback function
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons)
    .on(context.collect())  # Collect context chunks into a list
    .on_output()
)

# Relevance feedback functions
f_answer_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)  # Aggregate relevance scores using mean
)

# %%
# Initialize the TruLlama query engine recorder with feedback functions
tru_query_engine_recorder = TruLlama(
    query_engine,
    app_id="LlamaIndex_App1",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

#%%
rag_dataset = None
with open("./data/HistoryOfAlexnetDataset/rag_dataset.json","r") as f:
    rag_dataset = json.load(f)

rag_dataset['examples']
# %%
# Load the test dataset
# testset = pd.read_csv("testset.csv")

# # Create a dictionary to hold test questions and ground truths
# testset_dict = {
#     "question": list(testset["question"]),
#     "ground_truth": list(testset["ground_truth"]),
# }

# %%
# Query the engine with the first question in the testset and record the process
with tru_query_engine_recorder as recording:
    query_engine.query(rag_dataset['examples'][0]['query'])
    
    # %%
    # Retrieve the record of the app invocation
    rec = recording.get()  # Use .get if only one record
    # recs = recording.records  # Use .records if multiple

    # %%
    # Initialize Tru for accessing records and feedback
    tru = Tru()
    # Uncomment the following line to run the Tru dashboard
    # tru.run_dashboard()

    # %%
    # Retrieve the feedback results and print them
    for feedback, feedback_result in rec.wait_for_feedback_results().items():
        print(feedback.name, feedback_result.result)

    # Retrieve records and feedback for the specified app_id
    records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])

    # Display the records
    print(records.head())

    # %%
    # Run the Tru dashboard (uncomment to run)
    tru.run_dashboard()