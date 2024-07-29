#%%
import pandas as pd
# %%
llama3 = pd.read_csv("ragas_evaluation_llama3_as_eval.csv")
# %%
llama3_1 = pd.read_csv("ragas_evaluation_llama3.1_as_eval.csv")
# %%
metrics = ['faithfulness', 'answer_relevancy', 'context_precision',
       'context_recall', 'harmfulness']
# %%
llama3_metrics = llama3[metrics]
llama3_1_metrics = llama3_1[metrics]

#%%
print('-'*50)
print("llama3 Metrics:")
print(llama3_metrics.mean())

# %%
print('-'*50)
print("llama3.1 Metrics:")
print(llama3_1_metrics.mean())
print('-'*50)