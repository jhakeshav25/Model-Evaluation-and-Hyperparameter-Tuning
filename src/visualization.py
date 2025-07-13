import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(df):
    melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Score", hue="Metric", data=melted)
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
