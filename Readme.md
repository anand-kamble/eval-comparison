# README

## Overview

This repository contains Python scripts to evaluate and compare the performance of query engines built with the `llama_index` library using various metrics from the `ragas` library. The evaluation process involves setting up embeddings, loading documents, building a vector index, and constructing a query engine. The results are then compared using another script.

## Usage

### Evaluation Script

1. Load Environment Variables:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

2. Set Up Embeddings:
   ```python
   embeddings = OllamaEmbedding(model_name="llama3.1", base_url="http://class02:11434")
   Settings.embed_model = embeddings
   ```

3. Load Documents:
   ```python
   documents = SimpleDirectoryReader("./data/EvaluatingLlmSurveyPaperDataset", required_exts=[".pdf"], recursive=True).load_data()
   ```

4. Build Vector Index:
   ```python
   vector_index = VectorStoreIndex.from_documents(documents[:2])
   ```

5. Build Query Engine:
   ```python
   generator_llm = Ollama(model="llama3.1", request_timeout=600.0, base_url="http://class02:11434", additional_kwargs={"max_length": 512})
   query_engine = vector_index.as_query_engine(llm=generator_llm)
   ```

6. Prepare Test Set:
   ```python
   with open("data/EvaluatingLlmSurveyPaperDataset/rag_dataset.json", "r") as f:
       llama_rag_dataset = json.load(f)

   testset = {
       "question": [],
       "ground_truth": [],
   }

   for item in llama_rag_dataset["examples"]:
       testset["question"].append(item["query"])
       testset["ground_truth"].append(item["reference_answer"])

   dataset = Dataset.from_dict(testset)
   ```

7. Evaluate Query Engine:
   ```python
   result = evaluate(
       query_engine=query_engine,
       metrics=[faithfulness, answer_relevancy, context_precision, context_recall, harmfulness],
       dataset=testset,
       llm=Ollama(model="llama3.1", base_url="http://class03:11434"),
       raise_exceptions=False
   )
   ```

8. Save Evaluation Results:
   ```python
   result.to_pandas().to_csv("ragas_evaluation_llama3.1_as_eval.csv")
   ```

9. Save Timing Results:
   ```python
   with open("timing_results_llama3.1_as_eval.txt", "w") as f:
       for key, value in time_dict.items():
           f.write(f"{key}: {value} seconds\n")
   ```

### Comparison Script

This script is used to compare the evaluation results of different versions of the model.

1. Load Evaluation Results:
   ```python
   import pandas as pd

   llama3 = pd.read_csv("ragas_evaluation_llama3_as_eval.csv")
   llama3_1 = pd.read_csv("ragas_evaluation_llama3.1_as_eval.csv")
   ```

2. Define Metrics:
   ```python
   metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'harmfulness']
   ```

3. Extract Metrics:
   ```python
   llama3_metrics = llama3[metrics]
   llama3_1_metrics = llama3_1[metrics]
   ```

4. Print Metrics:
   ```python
   print('-'*50)
   print("llama3 Metrics:")
   print(llama3_metrics.mean())
   print('-'*50)
   print("llama3.1 Metrics:")
   print(llama3_1_metrics.mean())
   print('-'*50)
   ```

## Results

### Timing Results
- **embedding_setup**: 0.38918089866638184 seconds
- **document_loading**: 1.7253758907318115 seconds
- **vector_index_building**: 48.676960706710815 seconds
- **query_engine_building**: 0.0003561973571777344 seconds
- **testset_loading**: 0.16877079010009766 seconds
- **evaluation**: 11450.569771051407 seconds

### Comparison Results
--------------------------------------------------
**llama3 Metrics**:
- faithfulness: 0.791667
- answer_relevancy: 0.327564
- context_precision: 0.732143
- context_recall: 0.795402
- harmfulness: 0.379310

--------------------------------------------------
**llama3.1 Metrics**:
- faithfulness: 0.516667
- answer_relevancy: NaN
- context_precision: 0.718750
- context_recall: 0.797037
- harmfulness: 0.361345

--------------------------------------------------

