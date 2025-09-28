import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('enhanced_fever_medicine_recommendation.csv')

plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='Recommended_Medication', y='Temperature')
plt.title('Distribuzione della temperatura corporea per farmaco raccomandato')
plt.xlabel('Farmaco raccomandato')
plt.ylabel('Temperatura corporea (°C)')
plt.tight_layout()
plt.savefig('boxplot_temp_farmaco.png')
plt.close()