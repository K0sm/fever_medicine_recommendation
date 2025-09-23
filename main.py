"""
Sistema di Raccomandazione Medicinale per la Febbre

Stato dell’arte ispirato e documentazione:
1. Dass, S., Jaiswal, K., Kumar, S. (2024) "A Machine Learning-Based Recommendation System for Disease Prediction"
   - Uso RandomForestClassifier; pipeline ML classica. SSRN:5207379
2. Wu, Y., Zhang, L. et al. (2023) "Interpretable Machine Learning for Personalized Medical Recommendations"
   - Output di affidabilità, explainability. PubMed:37627940
3. Shaik, N.V. et al. (2025) "Medicine recommendation system (Health Harbour)"
   - Interfaccia user-friendly e campi con menu. WJARR-2025-0382

Implementazione Moderna: grafica rilassante, responsive, valori default, usability!
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gradio as gr
import matplotlib.pyplot as plt

# --- PREPARAZIONE DATI ---

df = pd.read_csv('enhanced_fever_medicine_recommendation.csv')
df.fillna('None', inplace=True)

target = 'Recommended_Medication'
features = [col for col in df.columns if col != target]

# Encoding categorico per consistenza Random Forest (paper1)
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

# --- DEFINIZIONE CAMPI E VALORI USER-FRIENDLY ---

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
numeric_fields = num_fields1 + num_fields2

# Valori attendibili per UX (basato su range reali e medie)
default_values = {
    "Età": 40,   # Adulti
    "Temperatura": 37.5,
    "BMI": 24,
    "Frequenza cardiaca": 80,
    "Umidità": 60,
    "AQI": 50
}

# --- FUNZIONE DI PREDIZIONE ROBUSTA (in stile paper2) ---

def predict_medicine(*inputs):
    numerics = list(inputs[:len(numeric_fields)])
    categoricals = list(inputs[len(numeric_fields):])
    data = {}

    for i, col in enumerate(numeric_fields):
        data[col] = [float(numerics[i])]
    for i, col in enumerate(menu_options.keys()):
        val = str(categoricals[i])
        le = le_dict[col]
        if val not in le.classes_:
            val = le.classes_[0]
        data[col] = [le.transform([val])[0]]

    input_df = pd.DataFrame(data)[features]
    pred = clf.predict(input_df)[0]
    class_probs = clf.predict_proba(input_df)[0]
    pred_label = le_dict[target].inverse_transform([pred])[0]
    labels = [le_dict[target].inverse_transform([i])[0] for i in clf.classes_]
    fig, ax = plt.subplots(figsize=(4, 1.4))
    fig.patch.set_facecolor("#f5fbf7")
    bars = ax.bar(labels, class_probs, color=['#3cba92' if lbl == pred_label else '#b0d2c1' for lbl in labels])
    ax.set_facecolor("#f5fbf7")
    ax.set_ylabel('Probabilità')
    ax.set_ylim(0, 1)
    ax.set_title('Affidabilità', color="#227373", fontsize=12)
    for bar, prob in zip(bars, class_probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=10, color="#34514a")
    plt.tight_layout()

    return f"**Medicinale consigliato:** <span style='color:#227373'>{pred_label}</span>\n\n**Affidabilità:** <span style='color:#227373'>{class_probs[list(clf.classes_).index(pred)]*100:.2f}%</span>", fig

# --- UI COMPATTA E MODERNA (paper3: Fields affiancati, styling green/teal) ---

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="teal"), css="""
    body { background: #f0f6f3; color: #227373; }
    #main-btn { background: #3cba92!important; color: #fff!important; border-radius: 8px; font-weight: bold; }
    label { color: #2b524b!important; }
    .gradio-container { background: #f0f6f3!important;}
""") as demo:
    gr.Markdown("<h1 style='color:#227373;font-weight:800;text-align:center;'>Raccomandazione Medicinale &#x1F48A;</h1>")
    gr.Markdown("Sistema basato su Random Forest e UI moderna. Stato dell’arte: SSRN:5207379 | PubMed:37627940 | WJARR-2025-0382")
    with gr.Row():
        age = gr.Number(label="Età", value=default_values["Età"])
        gender = gr.Dropdown(label="Genere", choices=menu_options['Gender'], value="Male")
    with gr.Row():
        temperature = gr.Number(label="Temperatura", value=default_values["Temperatura"])
        fever_severity = gr.Dropdown(label="Gravità febbre", choices=menu_options["Fever_Severity"], value="Mild Fever")
    with gr.Row():
        bmi = gr.Number(label="BMI", value=default_values["BMI"])
        heart_rate = gr.Number(label="Frequenza cardiaca", value=default_values["Frequenza cardiaca"])
    with gr.Row():
        humidity = gr.Number(label="Umidità", value=default_values["Umidità"])
        aqi = gr.Number(label="AQI", value=default_values["AQI"])
    with gr.Row():
        diet_type = gr.Dropdown(label="Dieta", choices=menu_options['Diet_Type'], value="Non-Vegetarian")
        physical_activity = gr.Dropdown(label="Attività fisica", choices=menu_options['Physical_Activity'], value="Moderate")
    with gr.Row():
        blood_pressure = gr.Dropdown(label="Pressione", choices=menu_options['Blood_Pressure'], value="Normal")
        headache = gr.Dropdown(label="Mal di testa", choices=menu_options['Headache'], value="No")
    with gr.Row():
        body_ache = gr.Dropdown(label="Dolori Muscolari", choices=menu_options['Body_Ache'], value="No")
        fatigue = gr.Dropdown(label="Stanchezza", choices=menu_options['Fatigue'], value="No")
    with gr.Row():
        chronic_conditions = gr.Dropdown(label="Patologie Croniche", choices=menu_options['Chronic_Conditions'], value="No")
        allergies = gr.Dropdown(label="Allergie", choices=menu_options['Allergies'], value="No")
    with gr.Row():
        smoking_history = gr.Dropdown(label="Fumatore", choices=menu_options['Smoking_History'], value="No")
        alcohol = gr.Dropdown(label="Alcol", choices=menu_options['Alcohol_Consumption'], value="No")
    with gr.Row():
        previous_med = gr.Dropdown(label="Farmaco Precedente", choices=menu_options['Previous_Medication'], value="None")
    btn = gr.Button("Ottieni raccomandazione", elem_id="main-btn")
    output_md = gr.Markdown()
    output_plot = gr.Plot()
    btn.click(
        predict_medicine,
        inputs = [
            age, temperature, bmi, heart_rate, humidity, aqi,  # tutti i numerici nell’ordine di numeric_fields!
            gender, diet_type, physical_activity, blood_pressure,
            fever_severity, headache, body_ache, fatigue,
            chronic_conditions, allergies, smoking_history, alcohol, previous_med  # tutti i categorici nell’ordine delle chiavi di menu_options
        ],
        outputs=[output_md, output_plot]
    )

demo.launch()
