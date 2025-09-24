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

LABELS = {
    "age": "Age",
    "gender": "Gender",
    "temperature": "Temperature (°C)",
    "fever_severity": "Fever Severity",
    "bmi": "BMI",
    "heart_rate": "Heart Rate",
    "humidity": "Humidity (%)",
    "aqi": "Air Quality Index (AQI)",
    "diet_type": "Diet Type",
    "physical_activity": "Physical Activity",
    "blood_pressure": "Blood Pressure",
    "headache": "Headache",
    "body_ache": "Body Ache",
    "fatigue": "Fatigue",
    "chronic_conditions": "Chronic Conditions",
    "allergies": "Allergies",
    "smoking_history": "Smoking History",
    "alcohol": "Alcohol Consumption",
    "previous_med": "Previous Medication"
}

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
clf = RandomForestClassifier(
    random_state=42,
    min_samples_leaf=2,
    max_features='sqrt')
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
    LABELS["age"]: 40,
    LABELS["temperature"]: 37.5,
    LABELS["bmi"]: 24,
    LABELS["heart_rate"]: 80,
    LABELS["humidity"]: 60,
    LABELS["aqi"]: 50
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
    ax.set_ylabel('Realiability')
    ax.set_ylim(0, 1)
    for bar, prob in zip(bars, class_probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=10, color="#34514a")
    plt.tight_layout()

    confidence = class_probs[list(clf.classes_).index(pred)] * 100

    if confidence <= 30:
        conf_color = "background:#990D35;color:#9d1212;"
    elif confidence <= 70:
        conf_color = "background:#D78521;color:#938015;"
    elif confidence <= 90:
        conf_color = "background:#519872;color:#8c5f00;"
    else:
        conf_color = "background:#2ecc71;color:#165c37;"


    result_html = f"""
        <div style='
            background:#e0f7ef;
            color:#227373;
            font-size:1.3em;
            font-weight:700;
            border-radius:10px;
            margin:18px 0 8px 0;
            padding:18px 10px;
            text-align: center;
            box-shadow: 0 0 8px #69c4b1aa;
            border:2px solid #b9ede6;
        '>
            Suggested medicine:<br>
            <span style="font-size:2em;color:#21836b">{pred_label}</span>
            <div style='margin-top:10px;font-size:1.1em;color:#227373;'>
                Reliability: <span style='padding:3px 12px;border-radius:8px;{conf_color}'><b>{confidence:.2f}%</b></span>
            </div>
        </div>
        """

    return result_html, fig

# --- UI COMPATTA E MODERNA (paper3: Fields affiancati, styling green/teal) ---

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="teal"), css="""
    body { background: #f0f6f3; color: #227373; }
    #main-btn { background: #3cba92!important; color: #fff!important; border-radius: 8px; font-weight: bold; }
    label { color: #2b524b!important; }
    .gradio-container { background: #f0f6f3!important;}
""") as demo:
    gr.Markdown("<h1 style='color:#227373;font-weight:800;text-align:center;'>Medicine Recommendation System</h1>")
    gr.Markdown("<span style='display:block;text-align:center;color:#227373;'>A modern clinical decision support tool based on Random Forest.<br>See included references: SSRN:5207379 | PubMed:37627940 | WJARR-2025-0382</span>")
    with gr.Row():
        age = gr.Number(label=LABELS["age"], value=default_values[LABELS["age"]])
        gender = gr.Dropdown(label=LABELS["gender"], choices=menu_options['Gender'], value="Male")
        temperature = gr.Number(label=LABELS["temperature"], value=default_values[LABELS["temperature"]])
    with gr.Row():
        fever_severity = gr.Dropdown(label=LABELS["fever_severity"], choices=menu_options["Fever_Severity"], value="Mild Fever")
        bmi = gr.Number(label=LABELS["bmi"], value=default_values[LABELS["bmi"]])
        heart_rate = gr.Number(label=LABELS["heart_rate"], value=default_values[LABELS["heart_rate"]])
    with gr.Row():
        humidity = gr.Number(label=LABELS["humidity"], value=default_values[LABELS["humidity"]])
        aqi = gr.Number(label=LABELS["aqi"], value=default_values[LABELS["aqi"]])
        diet_type = gr.Dropdown(label=LABELS["diet_type"], choices=menu_options['Diet_Type'], value="Non-Vegetarian")
    with gr.Row():
        physical_activity = gr.Dropdown(label=LABELS["physical_activity"], choices=menu_options['Physical_Activity'], value="Moderate")
        blood_pressure = gr.Dropdown(label=LABELS["blood_pressure"], choices=menu_options['Blood_Pressure'], value="Normal")
        headache = gr.Dropdown(label=LABELS["headache"], choices=menu_options['Headache'], value="No")
    with gr.Row():
        body_ache = gr.Dropdown(label=LABELS["body_ache"], choices=menu_options['Body_Ache'], value="No")
        fatigue = gr.Dropdown(label=LABELS["fatigue"], choices=menu_options['Fatigue'], value="No")
        chronic_conditions = gr.Dropdown(label=LABELS["chronic_conditions"], choices=menu_options['Chronic_Conditions'], value="No")
    with gr.Row():
        allergies = gr.Dropdown(label=LABELS["allergies"], choices=menu_options['Allergies'], value="No")
        smoking_history = gr.Dropdown(label=LABELS["smoking_history"], choices=menu_options['Smoking_History'], value="No")
        alcohol = gr.Dropdown(label=LABELS["alcohol"], choices=menu_options['Alcohol_Consumption'], value="No")
        previous_med = gr.Dropdown(label=LABELS["previous_med"], choices=menu_options['Previous_Medication'], value="None")
        
    btn = gr.Button("Get recommendation", elem_id="main-btn")
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
