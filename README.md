# Progetto: Sistema di Raccomandazione Medicinale per la Febbre
## Descrizione
Questo progetto sviluppa un sistema automatizzato in Python per raccomandare il medicinale più adatto a partire da sintomi, caratteristiche individuali e parametri clinici. L'interfaccia è presentata tramite Gradio per un'esperienza utente semplice e interattiva.

Il progetto include:
- pre-processing dei dati con rimozione degli outlier;
- encoding delle variabili categoriche per il modello;
- addestramento e valutazione di un classificatore Random Forest;
- interfaccia utente moderna e intuitiva sviluppata con Gradio, per facilitare l’interazione e la visualizzazione dei risultati;
- visualizzazione della probabilità di affidabilità delle raccomandazioni tramite grafici.

## Obiettivo
Fornire uno strumento di supporto alle decisioni cliniche che permetta una raccomandazione affidabile di farmaci, quali Ibuprofene e Paracetamolo, basata su dati sintetici e reali, migliorando l’efficacia e la personalizzazione delle terapie per pazienti con febbre.

## Requisiti
- Python 3.8 o superiore.
- Librerie Python: pandas, numpy, scikit-learn, matplotlib, gradio.

## Istruzioni per l’installazione
1. Clona o scarica il repository.
2. Installa le librerie necessarie: `pip install pandas numpy scikit-learn matplotlib gradio`.
3. Assicurati di avere il file `enhanced_fever_medicine_recommendation.csv` nella stessa cartella dello script.

## Avvio del progetto
Per avviare l’interfaccia utente, esegui lo script Python principale: `python main.py`.
Si aprirà una pagina web locale con il modulo per inserire i dati clinici del paziente (per comodità e testing ho preparato alcuni valori predefiniti).

## Utilizzo
- Inserisci i valori numerici (es. età, temperatura, BMI, parametri ambientali) usando i campi dedicati.
- Seleziona le opzioni categoriche (es. genere, severità febbre, abitudini di vita) dai menu a tendina.
- Clicca sul pulsante "Get recommendation" per ottenere la raccomandazione del medicinale più adatto insieme alla sua probabilità di affidabilità visualizzata in un grafico.

## Riferimenti bibliografici
- Dass, S. et al. (2024) A Machine Learning-Based Recommendation System for Disease Prediction (SSRN:5207379)
- Wu, Y. et al. (2023) Interpretable Machine Learning for Personalized Medical Recommendations (PubMed:37627940)
- Shaik, N.V. et al. (2025) Medicine recommendation system (Health Harbour) (WJARR-2025-0382)

---

Progetto sviluppato per l’esame di Sistemi Multimediali, Università degli Studi di Bari.