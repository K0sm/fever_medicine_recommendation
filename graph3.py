import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('enhanced_fever_medicine_recommendation.csv')
df.columns = [c.strip() for c in df.columns]  # elimina spazi

print(df.columns)  # Per controllare i nomi veri

plt.figure(figsize=(8,6))
sns.violinplot(
    data=df,
    x='Fever_Severity',
    y='Age',
    hue='Recommended_Medication',
    split=True
)
plt.title("Distribuzione dell'età per gravità della febbre e farmaco raccomandato")
plt.xlabel('Gravità della febbre')
plt.ylabel('Età')
plt.legend(title='Farmaco raccomandato')
plt.tight_layout()
plt.savefig('violinplot_eta_febbre_farmaco.png')
plt.close()
