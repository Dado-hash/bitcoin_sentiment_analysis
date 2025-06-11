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

def check_dataset_exists(filename):
    """
    Verifica se un dataset esiste e mostra info di base
    """
    if not os.path.exists(filename):
        return False
    
    try:
        # Prova diverse codifiche
        encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filename, sep=';', encoding=encoding, nrows=5)  # Solo prime 5 righe per test
                print(f"✓ Dataset trovato: {filename} (encoding: {encoding})")
                break
            except:
                continue
        
        if df is None:
            try:
                df = pd.read_csv(filename, nrows=5)  # Prova con separatore standard
                print(f"✓ Dataset trovato: {filename} (formato standard)")
            except:
                print(f"✗ Errore nel leggere {filename}")
                return False
        
        print(f"  Colonne: {list(df.columns)}")
        print(f"  Prime righe visualizzate correttamente")
        return True
        
    except Exception as e:
        print(f"✗ Errore nell'analizzare {filename}: {e}")
        return False

def test_bitcoin_analysis(dataset_filename="Dataset tesi.csv"):
    """
    Test completo del sistema di analisi Bitcoin
    """
    print("=== INIZIO TEST SISTEMA ANALISI BITCOIN ===\n")
    
    # Verifica esistenza dataset
    print(f"Verifica dataset: {dataset_filename}")
    if not check_dataset_exists(dataset_filename):
        print(f"✗ Dataset {dataset_filename} non trovato o non leggibile")
        return False
    
    # Importa e testa il modulo principale
    print("\n1. Test del modulo di analisi...")
    
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
    
    # Test caricamento dati
    print(f"\n2. Test caricamento dati da {dataset_filename}...")
    df = load_and_clean_data(dataset_filename)
    
    if df is None:
        print("✗ Caricamento dati fallito")
        return False
    
    print(f"✓ Dati caricati: {len(df)} righe")
    print(f"✓ Colonne: {list(df.columns)}")
    
    # Test analisi sentiment
    print("\n3. Test analisi sentiment...")
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
    
    # Test creazione features
    print("\n4. Test creazione features...")
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
    
    # Test training modello
    print("\n5. Test training modello...")
    try:
        model, df_with_predictions = train_regression_model(df)
        
        if model is None:
            print("✗ Modello non addestrato")
            return False
            
        print("✓ Modello addestrato correttamente")
        
    except Exception as e:
        print(f"✗ Errore training modello: {e}")
        return False
    
    # Test visualizzazioni
    print("\n6. Test visualizzazioni...")
    try:
        create_visualizations(df)
        print("✓ Visualizzazioni create")
        
    except Exception as e:
        print(f"✗ Errore visualizzazioni: {e}")
        return False
    
    # Verifica output file
    print("\n7. Verifica file output...")
    if os.path.exists("risultati_analisi.csv"):
        results_df = pd.read_csv("risultati_analisi.csv")
        print(f"✓ File risultati salvato: {len(results_df)} righe")
    else:
        print("! File risultati non trovato (possibile errore nel salvataggio)")
    
    print("\n=== TEST COMPLETATO CON SUCCESSO ===")
    print(f"\nDataset utilizzato: {dataset_filename}")
    print("File output generati:")
    print("- risultati_analisi.csv (risultati analisi)")
    
    return True

def quick_test_dataset_only(dataset_filename):
    """
    Test rapido solo per verificare la struttura del dataset
    """
    print(f"=== TEST RAPIDO DATASET: {dataset_filename} ===\n")
    
    if not os.path.exists(dataset_filename):
        print(f"✗ File {dataset_filename} non trovato")
        return False
    
    # Test caricamento con diverse configurazioni
    configurations = [
        {'sep': ';', 'encoding': 'utf-8'},
        {'sep': ';', 'encoding': 'ISO-8859-1'},
        {'sep': ';', 'encoding': 'cp1252'},
        {'sep': ',', 'encoding': 'utf-8'},
        {'sep': ',', 'encoding': 'ISO-8859-1'},
    ]
    
    df = None
    used_config = None
    
    for config in configurations:
        try:
            df = pd.read_csv(dataset_filename, **config)
            used_config = config
            print(f"✓ Dataset caricato con: {config}")
            break
        except Exception as e:
            continue
    
    if df is None:
        print("✗ Impossibile caricare il dataset con nessuna configurazione")
        return False
    
    print(f"\nInfo dataset:")
    print(f"- Righe: {len(df)}")
    print(f"- Colonne: {list(df.columns)}")
    print(f"- Dimensioni: {df.shape}")
    
    # Mostra prime righe
    print(f"\nPrime 3 righe:")
    print(df.head(3).to_string())
    
    # Verifica colonne attese
    expected_columns = ['timestamp', 'text', 'btc_price']
    missing_cols = [col for col in expected_columns if col not in df.columns]
    
    if missing_cols:
        print(f"\n⚠️  Colonne mancanti: {missing_cols}")
        print("Colonne presenti nel dataset:", list(df.columns))
        print("Assicurati che il dataset abbia le colonne: timestamp, text, btc_price")
    else:
        print("\n✓ Tutte le colonne attese sono presenti")
        
        # Test parsing delle colonne
        try:
            # Test timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
                print(f"✓ Timestamp parsato: range {df['timestamp'].min()} - {df['timestamp'].max()}")
            
            # Test prezzo
            if 'btc_price' in df.columns:
                # Prova diversi formati
                df['btc_price'] = df['btc_price'].astype(str).str.replace(',', '.', regex=False)
                df['btc_price'] = pd.to_numeric(df['btc_price'], errors='coerce')
                
                valid_prices = df['btc_price'].dropna()
                if len(valid_prices) > 0:
                    print(f"✓ Prezzi parsati: range €{valid_prices.min():.2f} - €{valid_prices.max():.2f}")
                else:
                    print("⚠️  Nessun prezzo valido trovato")
            
            # Test testi
            if 'text' in df.columns:
                text_lengths = df['text'].astype(str).str.len()
                print(f"✓ Testi: lunghezza media {text_lengths.mean():.1f} caratteri")
                
        except Exception as e:
            print(f"⚠️  Errore nel parsing: {e}")
    
    # Salva versione pulita per test
    try:
        df.to_csv("test_cleaned_dataset.csv", index=False, **used_config)
        print(f"\n✓ Dataset pulito salvato come: test_cleaned_dataset.csv")
    except:
        print("\n⚠️  Non è stato possibile salvare il dataset pulito")
    
    print(f"\n=== TEST DATASET COMPLETATO ===")
    return True

if __name__ == "__main__":
    print("=== SISTEMA TEST ANALISI BITCOIN ===")
    print("\nOpzioni disponibili:")
    print("1. Test completo con dataset esistente")
    print("2. Genera nuovo dataset di test")
    print("3. Test rapido struttura dataset")
    print("4. Genera dataset + Test completo")
    
    choice = input("\nInserisci la tua scelta (1-4): ").strip()
    
    if choice == "1":
        # Test con dataset esistente
        dataset_name = input("Nome del dataset (default: 'Dataset tesi.csv'): ").strip()
        if not dataset_name:
            dataset_name = "Dataset tesi.csv"
        
        if os.path.exists(dataset_name):
            test_bitcoin_analysis(dataset_name)
        else:
            print(f"Dataset {dataset_name} non trovato!")
            create_new = input("Vuoi generare un dataset di test? (s/n): ").strip().lower()
            if create_new == 's':
                print("Generazione dataset di test...")
                generate_bitcoin_dataset(num_records=200, filename=dataset_name)
                test_bitcoin_analysis(dataset_name)
    
    elif choice == "2":
        # Solo generazione dataset
        num_records = input("Numero di record (default: 500): ").strip()
        if num_records.isdigit():
            num_records = int(num_records)
        else:
            num_records = 500
        
        filename = input("Nome file (default: 'Dataset tesi.csv'): ").strip()
        if not filename:
            filename = "Dataset tesi.csv"
        
        generate_bitcoin_dataset(num_records=num_records, filename=filename)
    
    elif choice == "3":
        # Test rapido dataset
        dataset_name = input("Nome del dataset da testare (default: 'Dataset tesi.csv'): ").strip()
        if not dataset_name:
            dataset_name = "Dataset tesi.csv"
        
        quick_test_dataset_only(dataset_name)
    
    elif choice == "4":
        # Genera + Test completo
        print("Generazione dataset di test...")
        generate_bitcoin_dataset(num_records=200, filename="Dataset tesi.csv")
        test_bitcoin_analysis("Dataset tesi.csv")
    
    else:
        print("Scelta non valida!")
        print("Eseguo test rapido con dataset predefinito...")
        if os.path.exists("Dataset tesi.csv"):
            quick_test_dataset_only("Dataset tesi.csv")
        else:
            print("Nessun dataset trovato. Generazione dataset di test...")
            generate_bitcoin_dataset(num_records=100, filename="Dataset tesi.csv")
            quick_test_dataset_only("Dataset tesi.csv")