import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('enhanced_fever_medicine_recommendation.csv')
df.columns = [c.strip() for c in df.columns]

plt.figure(figsize=(8,6))
sns.countplot(data=df, x='Previous_Medication', hue='Recommended_Medication')
plt.title('Influenza del farmaco già assunto sulla raccomandazione finale')
plt.xlabel('Farmaco assunto in precedenza')
plt.ylabel('Conteggio raccomandazione')
plt.legend(title='Farmaco raccomandato')
plt.tight_layout()
plt.savefig('barplot_storico_raccomandazione.png')
plt.close()
