import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('enhanced_fever_medicine_recommendation.csv')
df.fillna('None', inplace=True)

target = 'Recommended_Medication'
features = [col for col in df.columns if col != target]

# Label Encoding
le_dict = {}
for col in df.columns:
    if df[col].dtype == 'object' or col == 'Previous_Medication':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Menu e campi numerici
menu_options = {
    'Fever_Severity': ['Normal', 'Mild Fever', 'High Fever'],
    'Gender': ['Female', 'Male'],
    'Headache': ['No', 'Yes'],
    'Body_Ache': ['No', 'Yes'],
    'Fatigue': ['No', 'Yes'],
    'Chronic_Conditions': ['No', 'Yes'],
    'Allergies': ['No', 'Yes'],
    'Smoking_History': ['No', 'Yes'],
    'Alcohol_Consumption': ['No', 'Yes'],
    'Physical_Activity': ['Sedentary', 'Moderate', 'Active'],
    'Diet_Type': ['Vegan', 'Vegetarian', 'Non-Vegetarian'],
    'Blood_Pressure': ['Normal', 'High', 'Low'],
    'Previous_Medication': ['None', 'Ibuprofen', 'Paracetamol', 'Aspirin'],
}

num_fields1 = ['Age', 'BMI', 'Temperature', 'Heart_Rate']
num_fields2 = ['Humidity', 'AQI']
categorical_fields1 = [
    'Gender', 'Diet_Type', 'Physical_Activity', 'Blood_Pressure'
]
categorical_fields2 = [
    'Fever_Severity', 'Headache', 'Body_Ache', 'Fatigue',
    'Chronic_Conditions', 'Allergies', 'Smoking_History', 'Alcohol_Consumption', 'Previous_Medication'
]

def predict_medicine(*inputs):
    input_dict = dict(zip(num_fields1 + num_fields2 + categorical_fields1 + categorical_fields2, inputs))
    input_df = pd.DataFrame([input_dict])
    # encode
    for col in set(categorical_fields1 + categorical_fields2):
        if col in le_dict:
            input_df[col] = le_dict[col].transform(input_df[col].astype(str))
    for col in num_fields1 + num_fields2:
        input_df[col] = pd.to_numeric(input_df[col])
    pred = clf.predict(input_df[features])[0]
    class_probs = clf.predict_proba(input_df[features])[0]
    pred_label = le_dict[target].inverse_transform([pred])[0]
    labels = [le_dict[target].inverse_transform([i])[0] for i in clf.classes_]
    fig, ax = plt.subplots(figsize=(5,2))
    bars = ax.bar(labels, class_probs, color=['#2794eb' if lbl==pred_label else '#ccc' for lbl in labels])
    ax.set_ylabel('Probabilità')
    ax.set_ylim(0,1)
    ax.set_title('Affidabilità della Raccomandazione')
    for bar, prob in zip(bars, class_probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    return (
        f"### Medicinale consigliato\n**{pred_label}**\n\n### Affidabilità\n**{class_probs[list(clf.classes_).index(pred)]*100:.2f}%**",
        fig
    )

with gr.Blocks() as demo:
    gr.Markdown("# Raccomandazione Medicinale per Febbre")
    gr.Markdown("Inserisci i dati nelle schede qua sotto.")
    with gr.Row():
        with gr.Column():
            with gr.Tab("Anagrafica e Vitali"):
                age = gr.Number(label="Età")
                bmi = gr.Number(label="BMI")
                temperature = gr.Number(label="Temperatura corporea")
                hr = gr.Number(label="Heart Rate")
            with gr.Tab("Ambiente"):
                humidity = gr.Number(label="Umidità")
                aqi = gr.Number(label="AQI")
        with gr.Column():
            with gr.Tab("Profilo"):
                gender = gr.Dropdown(label="Genere", choices=menu_options['Gender'])
                diet = gr.Dropdown(label="Dieta", choices=menu_options['Diet_Type'])
                activity = gr.Dropdown(label="Attività fisica", choices=menu_options['Physical_Activity'])
                bp = gr.Dropdown(label="Pressione", choices=menu_options['Blood_Pressure'])
            with gr.Tab("Sintomi e Storico"):
                fever_sev = gr.Dropdown(label="Gravità febbre", choices=menu_options['Fever_Severity'])
                headache = gr.Dropdown(label="Mal di testa", choices=menu_options['Headache'])
                body_ache = gr.Dropdown(label="Dolori muscolari", choices=menu_options['Body_Ache'])
                fatigue = gr.Dropdown(label="Stanchezza", choices=menu_options['Fatigue'])
                chronic = gr.Dropdown(label="Patologie croniche", choices=menu_options['Chronic_Conditions'])
                allerg = gr.Dropdown(label="Allergie", choices=menu_options['Allergies'])
                smoke = gr.Dropdown(label="Fumatore", choices=menu_options['Smoking_History'])
                alcohol = gr.Dropdown(label="Alcol", choices=menu_options['Alcohol_Consumption'])
                prev_med = gr.Dropdown(label="Farmaco precedente", choices=menu_options['Previous_Medication'])
    btn = gr.Button("Ottieni Raccomandazione")
    output_md = gr.Markdown()
    output_plot = gr.Plot()
    btn.click(
        predict_medicine,
        inputs=[
            age, bmi, temperature, hr, humidity, aqi,
            gender, diet, activity, bp,
            fever_sev, headache, body_ache, fatigue, chronic, allerg, smoke, alcohol, prev_med,
        ],
        outputs=[output_md, output_plot]
    )

demo.launch()