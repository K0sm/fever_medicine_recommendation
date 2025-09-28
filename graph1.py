import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

df = pd.read_csv('enhanced_fever_medicine_recommendation.csv')
plt.figure(figsize=(7,5))
sns.countplot(data=df, x='Recommended_Medication', hue='Gender')
plt.title('Distribuzione del medicinale raccomandato rispetto al genere')
plt.xlabel('Farmaco raccomandato')
plt.ylabel('Frequenza')
plt.legend(title='Sesso')
plt.tight_layout()
plt.savefig('medicine_distribution.png')
plt.close()