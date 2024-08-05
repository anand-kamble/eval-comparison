#%%
import pandas as pd
import matplotlib.pyplot as plt

QUERY_MODEL = "llama3"
EVALUATION_MODEL = "llama3.1"
DATASET = "PatronusAIFinanceBenchDataset"
# %%
# Query Model: llama3
# Evaluation Model: llama3
llama3 = pd.read_csv(f"results/{DATASET}_query_llama3_eval_llama3.csv")
# %%
# Query Model: llama3.1
# Evaluation Model: llama3.1
llama3_1 = pd.read_csv(f"results/{DATASET}_query_llama3.1_eval_llama3.1.csv")

#%%
llama3_1["faithfulness"].value_counts()

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




#%%
# Query Model: llama3
# Evaluation Model: llama3.1
llama3_llama3_1 = pd.read_csv(f"results/{DATASET}_query_llama3_eval_llama3.1.csv")

# Query Model: llama3.1
# Evaluation Model: llama3
llama3_1_llama3 = pd.read_csv(f"results/{DATASET}_query_llama3.1_eval_llama3.csv")
# %%
llama3_metrics = llama3[metrics].mean()
llama3_1_metrics = llama3_1[metrics].mean()
llama3_llama3_1_metrics = llama3_llama3_1[metrics].mean()
llama3_1_llama3_metrics = llama3_1_llama3[metrics].mean()

# Create a dataframe for plotting
data = pd.DataFrame({
    'Llama 3 (self-eval)': llama3_metrics,
    'Llama 3.1 (self-eval)': llama3_1_metrics,
    'Llama 3 (eval by Llama 3.1)': llama3_llama3_1_metrics,
    'Llama 3.1 (eval by Llama 3)': llama3_1_llama3_metrics
})

fig, ax = plt.subplots(figsize=(12, 8),dpi=200)

# Plotting the metrics
data.plot(kind='bar', ax=ax, rot=0)
ax.set_title('Comparison of Evaluation Metrics for Different Model Combinations')
ax.set_xlabel('Metrics')
ax.set_ylabel('Mean Values')
ax.legend(title='Model Evaluations', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)

plt.figtext(0.5, -0.05, 'Using Mini Truthful QA Dataset', ha='center', fontsize=12)

plt.tight_layout()
plt.show()
# plt.savefig('comparison_metrics.png')


# %%
