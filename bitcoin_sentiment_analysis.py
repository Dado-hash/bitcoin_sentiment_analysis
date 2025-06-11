import pandas as pd 
from datetime import timedelta
import tweetnlp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_and_clean_data(file_path):
    """Carica e pulisce il dataset"""
    try:
        # Carica il dataset
        df = pd.read_csv(file_path, sep=';', encoding='ISO-8859-1')
        df.columns = ['timestamp', 'text', 'btc_price']
        
        # Pulizia valori numerici del prezzo BTC
        df['btc_price'] = df['btc_price'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df['btc_price'] = pd.to_numeric(df['btc_price'], errors='coerce')
        
        # Conversione timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
        
        # Rimuovi righe con valori mancanti
        df = df.dropna()
        
        print(f"Dataset caricato: {len(df)} righe")
        return df
        
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        return None

def perform_sentiment_analysis(df):
    """Esegue l'analisi del sentiment sui tweet"""
    print("Iniziando analisi del sentiment...")
    
    # Inizializza il modello di sentiment
    model = tweetnlp.load_model('sentiment') 
    
    # Analizza il sentiment per ogni tweet
    sentiments = []
    for i, text in enumerate(df['text']):
        try:
            result = model.predict(text)
            sentiments.append(result['label'])
            
            if (i + 1) % 100 == 0:
                print(f"Processati {i + 1}/{len(df)} tweet")
                
        except Exception as e:
            print(f"Errore nell'analisi del tweet {i}: {e}")
            sentiments.append('neutral')  # Default fallback
    
    df['sentiment'] = sentiments
    
    # Mappatura a numeri: positive = 1, neutral = 0, negative = -1
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)
    
    print("Analisi del sentiment completata!")
    return df

def create_prediction_features(df, horizon_days=10):
    """Crea le features per la previsione del prezzo futuro"""
    # Ordina per timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Crea una colonna con il prezzo di BTC X giorni dopo
    df['future_price'] = df['btc_price'].shift(-horizon_days)
    
    # Calcola la variazione percentuale del prezzo
    df['price_change'] = ((df['future_price'] - df['btc_price']) / df['btc_price']) * 100
    
    # Crea features aggiuntive
    df['price_direction'] = (df['future_price'] > df['btc_price']).astype(int)
    
    # Rimuovi righe senza prezzo futuro
    df = df.dropna()
    
    print(f"Features create per {len(df)} osservazioni")
    return df

def train_regression_model(df):
    """Addestra il modello di regressione lineare"""
    # Caratteristiche e target
    X = df[['sentiment_score']]
    y = df['future_price']
    
    # Dividi in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modello di regressione lineare
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Previsioni
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metriche di valutazione
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    print(f"R² Training: {train_r2:.4f}")
    print(f"R² Test: {test_r2:.4f}")
    print(f"MSE Training: {train_mse:.2f}")
    print(f"MSE Test: {test_mse:.2f}")
    
    # Previsioni su tutto il dataset per il plot
    df['predicted_price'] = model.predict(X)
    
    return model, df

def create_visualizations(df):
    """Crea visualizzazioni dei risultati"""
    plt.style.use('seaborn-v0_8')
    
    # Figura con subplot multipli
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # 1. Confronto prezzo attuale vs previsione
    axes[0, 0].plot(df['timestamp'], df['btc_price'], label='Prezzo Attuale BTC', alpha=0.7)
    axes[0, 0].plot(df['timestamp'], df['predicted_price'], label='Previsione (+10gg)', linestyle='--', alpha=0.8)
    axes[0, 0].set_xlabel("Data")
    axes[0, 0].set_ylabel("Prezzo BTC (€)")
    axes[0, 0].set_title("Confronto Prezzo Attuale vs Previsione")
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Distribuzione del sentiment
    sentiment_counts = df['sentiment'].value_counts()
    axes[0, 1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title("Distribuzione del Sentiment")
    
    # 3. Sentiment vs Prezzo
    axes[1, 0].scatter(df['sentiment_score'], df['btc_price'], alpha=0.6, c=df['sentiment_score'], cmap='RdYlGn')
    axes[1, 0].set_xlabel("Sentiment Score")
    axes[1, 0].set_ylabel("Prezzo BTC (€)")
    axes[1, 0].set_title("Relazione Sentiment - Prezzo BTC")
    
    # 4. Prezzo attuale vs prezzo futuro
    axes[1, 1].scatter(df['btc_price'], df['future_price'], alpha=0.6)
    axes[1, 1].plot([df['btc_price'].min(), df['btc_price'].max()], 
                    [df['btc_price'].min(), df['btc_price'].max()], 'r--', alpha=0.8)
    axes[1, 1].set_xlabel("Prezzo Attuale (€)")
    axes[1, 1].set_ylabel("Prezzo Futuro (+10gg) (€)")
    axes[1, 1].set_title("Prezzo Attuale vs Prezzo Futuro")
    
    plt.tight_layout()
    plt.show()
    
    # Analisi aggiuntiva: correlazione sentiment-direzione prezzo
    correlation = df['sentiment_score'].corr(df['price_direction'])
    print(f"\nCorrelazione tra sentiment e direzione del prezzo: {correlation:.4f}")
    
    # Statistiche descrittive per sentiment
    print("\nStatistiche del sentiment per direzione del prezzo:")
    print(df.groupby('price_direction')['sentiment_score'].describe())

def main():
    """Funzione principale"""
    print("=== Analisi Sentiment e Previsione Prezzo Bitcoin ===\n")
    
    # 1. Carica e pulisce i dati
    df = load_and_clean_data("Dataset tesi.csv")
    if df is None:
        return
    
    # 2. Analisi del sentiment
    df = perform_sentiment_analysis(df)
    
    # 3. Crea features per la previsione
    df = create_prediction_features(df, horizon_days=10)
    
    # 4. Addestra il modello
    model, df = train_regression_model(df)
    
    # 5. Crea visualizzazioni
    create_visualizations(df)
    
    # 6. Salva risultati
    df.to_csv("risultati_analisi.csv", index=False)
    print("\nRisultati salvati in 'risultati_analisi.csv'")
    
    print("\n=== Analisi completata ===")

if __name__ == "__main__":
    main()