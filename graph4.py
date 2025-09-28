import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('enhanced_fever_medicine_recommendation.csv')
df.columns = [c.strip() for c in df.columns]

order_yesno = ['Yes', 'No']

# --- Grafico 1: Farmaco vs Condizioni Croniche ---
plt.figure(figsize=(7,6))
sns.countplot(
    data=df,
    x='Recommended_Medication',
    hue='Chronic_Conditions',
    hue_order=order_yesno
)
plt.title('Farmaco raccomandato vs Condizioni Croniche')
plt.xlabel('Farmaco raccomandato')
plt.ylabel('Frequenza')
plt.legend(title='Cond. Croniche', loc='best')
plt.tight_layout()
plt.savefig('barplot_condizioni_croniche.png')
plt.close()

# --- Grafico 2: Farmaco vs Allergie ---
plt.figure(figsize=(7,6))
sns.countplot(
    data=df,
    x='Recommended_Medication',
    hue='Allergies',
    hue_order=order_yesno
)
plt.title('Farmaco raccomandato vs Allergie')
plt.xlabel('Farmaco raccomandato')
plt.ylabel('Frequenza')
plt.legend(title='Allergie', loc='best')
plt.tight_layout()
plt.savefig('barplot_allergie.png')
plt.close()
