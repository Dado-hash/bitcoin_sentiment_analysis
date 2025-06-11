import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_bitcoin_dataset(num_records=500, filename="Dataset tesi.csv"):
    """
    Genera un dataset di test per l'analisi sentiment Bitcoin
    """
    
    # Imposta seed per riproducibilità
    np.random.seed(42)
    random.seed(42)
    
    # Genera date dal 2024-01-01 in poi
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i*2) for i in range(num_records)]
    
    # Genera prezzi Bitcoin realistici con trend e volatilità
    base_price = 45000
    prices = []
    current_price = base_price
    
    for i in range(num_records):
        # Trend generale + rumore + volatilità
        trend = 0.001 * i  # Trend leggermente crescente
        volatility = np.random.normal(0, 0.02)  # Volatilità 2%
        shock = np.random.choice([0, 0, 0, 0, 0.05, -0.05], p=[0.94, 0.01, 0.01, 0.01, 0.02, 0.01])  # Shock occasionali
        
        price_change = trend + volatility + shock
        current_price = current_price * (1 + price_change)
        
        # Mantieni prezzo realistico
        current_price = max(20000, min(80000, current_price))
        prices.append(round(current_price, 2))
    
    # Genera tweet con sentiment correlato al prezzo
    tweets = []
    sentiments = []
    
    # Template di tweet per diversi sentiment
    positive_templates = [
        "Bitcoin is going to the moon! 🚀 #BTC #crypto",
        "Great news for Bitcoin! Bull run incoming! 📈",
        "BTC breaking resistance levels! Time to buy! 💰",
        "Bitcoin adoption is accelerating! Bullish! 🔥",
        "Amazing Bitcoin performance today! 📊",
        "BTC hitting new highs! This is amazing! ⬆️",
        "Bitcoin showing strong momentum! 💪",
        "Incredible Bitcoin growth! Buy the dip! 🎯"
    ]
    
    negative_templates = [
        "Bitcoin is crashing hard! 📉 #BTC #crypto",
        "Bad news for Bitcoin! Bear market ahead! 📉",
        "BTC breaking support levels! Time to sell! 💸",
        "Bitcoin regulation concerns! Bearish! 🔻",
        "Terrible Bitcoin performance today! 📊",
        "BTC hitting new lows! This is concerning! ⬇️",
        "Bitcoin showing weak momentum! 📉",
        "Disappointing Bitcoin drop! Sell now! 🚨"
    ]
    
    neutral_templates = [
        "Bitcoin price is stable today #BTC #crypto",
        "BTC consolidating around current levels 📊",
        "Bitcoin market is quiet today 🤔",
        "Waiting for Bitcoin direction 📈📉",
        "BTC trading sideways today 📊",
        "Bitcoin price holding steady 🔄",
        "Mixed signals from Bitcoin market 📊",
        "Bitcoin volume is average today 📈"
    ]
    
    for i in range(num_records):
        # Correlazione sentiment-prezzo con del rumore
        if i > 0:
            price_change = (prices[i] - prices[i-1]) / prices[i-1]
            
            # Probabilità di sentiment basata su variazione prezzo
            if price_change > 0.02:  # +2%
                sentiment_probs = [0.7, 0.2, 0.1]  # [positive, neutral, negative]
            elif price_change < -0.02:  # -2%
                sentiment_probs = [0.1, 0.2, 0.7]
            else:
                sentiment_probs = [0.3, 0.4, 0.3]
        else:
            sentiment_probs = [0.33, 0.34, 0.33]
        
        # Seleziona sentiment
        sentiment = np.random.choice(['positive', 'neutral', 'negative'], p=sentiment_probs)
        sentiments.append(sentiment)
        
        # Seleziona tweet template
        if sentiment == 'positive':
            tweet = random.choice(positive_templates)
        elif sentiment == 'negative':
            tweet = random.choice(negative_templates)
        else:
            tweet = random.choice(neutral_templates)
        
        # Aggiungi variazione al tweet
        variations = [
            f"Just saw: {tweet}",
            f"RT @user: {tweet}",
            f"{tweet} What do you think?",
            f"IMO: {tweet}",
            tweet,
            tweet,
            tweet  # Peso maggiore per tweet originali
        ]
        
        final_tweet = random.choice(variations)
        tweets.append(final_tweet)
    
    # Crea DataFrame
    df = pd.DataFrame({
        'timestamp': [date.strftime('%d/%m/%Y %H:%M') for date in dates],
        'text': tweets,
        'btc_price': [f"{price:.2f}".replace('.', ',') for price in prices]  # Formato europeo
    })
    
    # Salva dataset
    df.to_csv(filename, sep=';', index=False, encoding='utf-8')
    
    print(f"Dataset generato: {filename}")
    print(f"Records: {len(df)}")
    print(f"Range temporale: {dates[0].strftime('%d/%m/%Y')} - {dates[-1].strftime('%d/%m/%Y')}")
    print(f"Range prezzi: €{min(prices):.2f} - €{max(prices):.2f}")
    print(f"Distribuzione sentiment attesa: {pd.Series(sentiments).value_counts().to_dict()}")
    
    return df

def test_bitcoin_analysis():
    """
    Test completo del sistema di analisi Bitcoin
    """
    print("=== INIZIO TEST SISTEMA ANALISI BITCOIN ===\n")
    
    # 1. Genera dataset di test
    print("1. Generazione dataset di test...")
    df_original = generate_bitcoin_dataset(num_records=200, filename="Dataset tesi.csv")
    
    # 2. Importa e testa il modulo principale
    print("\n2. Test del modulo di analisi...")
    
    try:
        # Importa le funzioni (assumendo che siano nel file corretto)
        from bitcoin_analysis import (
            load_and_clean_data, 
            perform_sentiment_analysis,
            create_prediction_features,
            train_regression_model,
            create_visualizations
        )
        
        print("✓ Moduli importati correttamente")
        
    except ImportError as e:
        print(f"✗ Errore import: {e}")
        print("Assicurati che il codice sia salvato come 'bitcoin_analysis.py'")
        return False
    
    # 3. Test caricamento dati
    print("\n3. Test caricamento dati...")
    df = load_and_clean_data("Dataset tesi.csv")
    
    if df is None:
        print("✗ Caricamento dati fallito")
        return False
    
    print(f"✓ Dati caricati: {len(df)} righe")
    print(f"✓ Colonne: {list(df.columns)}")
    
    # 4. Test analisi sentiment
    print("\n4. Test analisi sentiment...")
    try:
        df = perform_sentiment_analysis(df)
        
        if 'sentiment' not in df.columns:
            print("✗ Colonna sentiment mancante")
            return False
            
        sentiment_dist = df['sentiment'].value_counts()
        print(f"✓ Sentiment analizzato: {sentiment_dist.to_dict()}")
        
    except Exception as e:
        print(f"✗ Errore analisi sentiment: {e}")
        return False
    
    # 5. Test creazione features
    print("\n5. Test creazione features...")
    try:
        df = create_prediction_features(df, horizon_days=5)  # Riduci orizzonte per test
        
        expected_features = ['sentiment_score', 'price_change_1d']
        missing_features = [f for f in expected_features if f not in df.columns]
        
        if missing_features:
            print(f"✗ Features mancanti: {missing_features}")
            return False
            
        print(f"✓ Features create: {len(df)} osservazioni")
        
    except Exception as e:
        print(f"✗ Errore creazione features: {e}")
        return False
    
    # 6. Test training modello
    print("\n6. Test training modello...")
    try:
        model, df_with_predictions = train_regression_model(df)
        
        if model is None:
            print("✗ Modello non addestrato")
            return False
            
        print("✓ Modello addestrato correttamente")
        
    except Exception as e:
        print(f"✗ Errore training modello: {e}")
        return False
    
    # 7. Test visualizzazioni
    print("\n7. Test visualizzazioni...")
    try:
        create_visualizations(df)
        print("✓ Visualizzazioni create")
        
    except Exception as e:
        print(f"✗ Errore visualizzazioni: {e}")
        return False
    
    # 8. Verifica output file
    print("\n8. Verifica file output...")
    if os.path.exists("risultati_analisi.csv"):
        results_df = pd.read_csv("risultati_analisi.csv")
        print(f"✓ File risultati salvato: {len(results_df)} righe")
    else:
        print("! File risultati non trovato (possibile errore nel salvataggio)")
    
    print("\n=== TEST COMPLETATO CON SUCCESSO ===")
    print("\nRiepilogo file generati:")
    print("- Dataset tesi.csv (dataset di test)")
    print("- risultati_analisi.csv (risultati analisi)")
    
    return True

def quick_test_without_main():
    """
    Test rapido senza eseguire il main del modulo principale
    """
    print("=== TEST RAPIDO FUNZIONI INDIVIDUALI ===\n")
    
    # Genera dataset piccolo
    print("Generazione dataset di test...")
    generate_bitcoin_dataset(num_records=50, filename="test_dataset.csv")
    
    # Test manuale delle funzioni
    print("\nTest caricamento dati...")
    
    # Simula le funzioni principali
    try:
        df = pd.read_csv("test_dataset.csv", sep=';', encoding='ISO-8859-1')
        df.columns = ['timestamp', 'text', 'btc_price']
        
        # Pulizia dati
        df['btc_price'] = df['btc_price'].str.replace(',', '.', regex=False)
        df['btc_price'] = pd.to_numeric(df['btc_price'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
        df = df.dropna()
        
        print(f"✓ Dataset caricato: {len(df)} righe")
        print(f"✓ Range prezzi: €{df['btc_price'].min():.2f} - €{df['btc_price'].max():.2f}")
        
        # Test sentiment semplice
        positive_words = ['moon', 'bull', 'buy', 'great', 'amazing']
        negative_words = ['crash', 'bear', 'sell', 'bad', 'terrible']
        
        sentiments = []
        for text in df['text']:
            text_lower = str(text).lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiments.append('positive')
            elif neg_count > pos_count:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')
        
        df['sentiment'] = sentiments
        sentiment_dist = df['sentiment'].value_counts()
        print(f"✓ Sentiment distribuito: {sentiment_dist.to_dict()}")
        
        # Salva risultato test
        df.to_csv("test_results.csv", index=False)
        print("✓ File test_results.csv salvato")
        
        print("\n=== TEST RAPIDO COMPLETATO ===")
        return True
        
    except Exception as e:
        print(f"✗ Errore nel test: {e}")
        return False

if __name__ == "__main__":
    print("Scegli il tipo di test:")
    print("1. Test completo (richiede bitcoin_analysis.py)")
    print("2. Test rapido (solo generazione dataset)")
    
    choice = input("Inserisci 1 o 2: ").strip()
    
    if choice == "1":
        test_bitcoin_analysis()
    elif choice == "2":
        quick_test_without_main()
    else:
        print("Scelta non valida. Eseguo test rapido...")
        quick_test_without_main()