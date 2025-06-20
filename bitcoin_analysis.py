import pandas as pd 
from datetime import timedelta, datetime
import tweetnlp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import yfinance as yf  # Per scaricare dati Bitcoin reali
warnings.filterwarnings('ignore')

def download_bitcoin_prices(start_date, end_date):
    """Scarica i prezzi storici di Bitcoin da Yahoo Finance"""
    try:
        print(f"Scaricando prezzi Bitcoin dal {start_date} al {end_date}...")
        
        # Scarica dati Bitcoin
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
        
        if btc_data.empty:
            print("Nessun dato scaricato")
            return None
        
        # Prepara DataFrame
        btc_data = btc_data.reset_index()  # resetta l'indice, porta la colonna 'Date' nel dataframe
        btc_data.columns = ['_'.join(col).strip() for col in btc_data.columns.values]
        print(btc_data.columns)

        btc_prices = pd.DataFrame({
            'date': btc_data['Date_'],
            'price': btc_data['Close_BTC-USD']
        })
        
        print(f"Scaricati {len(btc_prices)} giorni di prezzi")
        print(f"Range: ${btc_prices['price'].min():.2f} - ${btc_prices['price'].max():.2f}")
        
        return btc_prices
        
    except Exception as e:
        print(f"Errore nel download: {e}")
        print("Assicurati di avere installato yfinance: pip install yfinance")
        return None

def load_and_clean_data(file_path):
    """Carica e pulisce il dataset"""
    try:
        # Prova diversi encoding e separatori
        encodings = ['utf-8', 'ISO-8859-1', 'cp1252']
        separators = [';', ',', '\t']
        
        df = None
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                    if len(df.columns) >= 3:
                        print(f"Dataset caricato con encoding={encoding}, sep='{sep}'")
                        break
                except:
                    continue
            if df is not None:
                break
        
        if df is None:
            raise Exception("Impossibile caricare il dataset")
        
        # Assegna nomi colonne
        if len(df.columns) >= 3:
            df.columns = ['timestamp', 'text', 'btc_price'] + list(df.columns[3:])
        
        # Pulizia timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        df['date'] = df['timestamp'].dt.date
        
        # Pulizia testo
        df['text'] = df['text'].astype(str)
        
        # Rimuovi righe invalide
        df = df.dropna(subset=['timestamp', 'text'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Dataset caricato: {len(df)} righe")
        print(f"Range temporale: {df['timestamp'].min()} - {df['timestamp'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Errore caricamento: {e}")
        return None

def perform_sentiment_analysis(df):
    """Analisi sentiment con fallback semplificato"""
    print("Analizzando sentiment...")
    
    try:
        # Prova tweetnlp
        model = tweetnlp.load_model('sentiment')
        use_tweetnlp = True
    except:
        print("tweetnlp non disponibile, uso analisi semplificata")
        use_tweetnlp = False
    
    sentiments = []
    
    if use_tweetnlp:
        for i, text in enumerate(df['text']):
            try:
                result = model.predict(str(text))
                sentiments.append(result['label'])
            except:
                sentiments.append('neutral')
            
            if (i + 1) % 100 == 0:
                print(f"Processati {i + 1}/{len(df)} tweet")
    else:
        # Analisi semplificata
        positive_words = ['good', 'great', 'buy', 'bull', 'moon', 'rise', 'up', 'profit', 'gain', 'bullish']
        negative_words = ['bad', 'sell', 'bear', 'crash', 'down', 'loss', 'drop', 'fall', 'bearish']
        
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
    
    # Mappa a score numerici
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)
    
    print(f"Sentiment analizzato: {df['sentiment'].value_counts().to_dict()}")
    return df

def create_daily_features(df):
    """Aggrega i dati per giorno e crea features"""
    # Aggrega per giorno
    daily_data = df.groupby('date').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'text': 'count'
    }).reset_index()
    
    # Flatten column names
    daily_data.columns = ['date', 'sentiment_mean', 'sentiment_std', 'sentiment_count', 'tweet_count']
    daily_data['sentiment_std'] = daily_data['sentiment_std'].fillna(0)
    
    # Aggiungi features temporali
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
    daily_data['is_weekend'] = daily_data['day_of_week'].isin([5, 6]).astype(int)
    
    # Media mobile del sentiment
    daily_data = daily_data.sort_values('date')
    daily_data['sentiment_ma_3'] = daily_data['sentiment_mean'].rolling(3, min_periods=1).mean()
    daily_data['sentiment_ma_7'] = daily_data['sentiment_mean'].rolling(7, min_periods=1).mean()
    
    print(f"Dati aggregati per {len(daily_data)} giorni")
    return daily_data

def merge_with_real_prices(daily_data, real_prices):
    """Unisce i dati sentiment con i prezzi reali di Bitcoin"""
    # Assicurati che le date siano dello stesso tipo
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    real_prices['date'] = pd.to_datetime(real_prices['date'])
    
    # Merge dei dati
    merged_data = pd.merge(daily_data, real_prices, on='date', how='inner')
    
    if len(merged_data) == 0:
        print("⚠️ Nessuna corrispondenza tra le date dei dati!")
        print(f"Range sentiment: {daily_data['date'].min()} - {daily_data['date'].max()}")
        print(f"Range prezzi: {real_prices['date'].min()} - {real_prices['date'].max()}")
        return None
    
    # Calcola variazioni di prezzo
    merged_data = merged_data.sort_values('date')
    merged_data['price_change_1d'] = merged_data['price'].pct_change(1)
    merged_data['price_change_3d'] = merged_data['price'].pct_change(3)
    merged_data['price_change_7d'] = merged_data['price'].pct_change(7)
    
    # Target: prezzo futuro (per validazione)
    merged_data['future_price_1d'] = merged_data['price'].shift(-1)
    merged_data['future_price_3d'] = merged_data['price'].shift(-3)
    merged_data['future_price_7d'] = merged_data['price'].shift(-7)
    
    # Calcola variazioni future
    merged_data['future_change_1d'] = (merged_data['future_price_1d'] - merged_data['price']) / merged_data['price']
    merged_data['future_change_3d'] = (merged_data['future_price_3d'] - merged_data['price']) / merged_data['price']
    merged_data['future_change_7d'] = (merged_data['future_price_7d'] - merged_data['price']) / merged_data['price']
    
    # Direzione del prezzo (su/giù)
    merged_data['future_direction_1d'] = (merged_data['future_change_1d'] > 0).astype(int)
    merged_data['future_direction_3d'] = (merged_data['future_change_3d'] > 0).astype(int)
    merged_data['future_direction_7d'] = (merged_data['future_change_7d'] > 0).astype(int)
    
    print(f"Dati uniti: {len(merged_data)} giorni con prezzi e sentiment")
    return merged_data

def train_prediction_model(data, prediction_horizon='1d'):
    """Addestra modello di previsione usando TimeSeriesSplit"""
    
    # Seleziona features e target
    feature_cols = ['sentiment_mean', 'sentiment_ma_3', 'sentiment_ma_7', 
                   'tweet_count', 'day_of_week', 'is_weekend']
    
    # Rimuovi features con tutti NaN
    available_features = [col for col in feature_cols if col in data.columns and not data[col].isna().all()]
    
    target_col = f'future_change_{prediction_horizon}'
    direction_col = f'future_direction_{prediction_horizon}'
    
    # Prepara dati
    df_clean = data.dropna(subset=available_features + [target_col])
    
    if len(df_clean) < 30:
        print(f"Dati insufficienti per previsione {prediction_horizon}: {len(df_clean)} giorni")
        return None, None
    
    X = df_clean[available_features]
    y_change = df_clean[target_col]  # Variazione percentuale
    y_direction = df_clean[direction_col]  # Direzione (0/1)
    
    # TimeSeriesSplit per validazione temporale corretta
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Metriche di validazione
    r2_scores = []
    direction_accuracies = []
    
    models = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_change.iloc[train_idx], y_change.iloc[test_idx]
        y_dir_train, y_dir_test = y_direction.iloc[train_idx], y_direction.iloc[test_idx]
        
        # Modello per variazione percentuale
        model = LinearRegression()
        model.fit(X_train, y_train)
        models.append(model)
        
        # Previsioni
        y_pred = model.predict(X_test)
        
        # Metriche
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
        
        # Accuratezza direzione
        direction_pred = (y_pred > 0).astype(int)
        direction_acc = (direction_pred == y_dir_test).mean()
        direction_accuracies.append(direction_acc)
    
    # Risultati medi
    avg_r2 = np.mean(r2_scores)
    avg_direction_acc = np.mean(direction_accuracies)
    
    print(f"\n=== Risultati Previsione {prediction_horizon} ===")
    print(f"R² medio: {avg_r2:.4f}")
    print(f"Accuratezza direzione: {avg_direction_acc:.4f} ({avg_direction_acc*100:.1f}%)")
    
    # Addestra modello finale su tutti i dati
    final_model = LinearRegression()
    final_model.fit(X, y_change)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'coefficient': final_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nImportanza features:")
    print(feature_importance)
    
    return final_model, df_clean

def create_visualizations(data):
    """Crea visualizzazioni complete"""
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Prezzo Bitcoin nel tempo
    axes[0, 0].plot(data['date'], data['price'], linewidth=2, color='orange')
    axes[0, 0].set_title('Prezzo Bitcoin nel Tempo', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Prezzo USD')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Sentiment medio nel tempo
    axes[0, 1].plot(data['date'], data['sentiment_mean'], color='green', alpha=0.7)
    axes[0, 1].plot(data['date'], data['sentiment_ma_7'], color='red', linewidth=2)
    axes[0, 1].set_title('Sentiment nel Tempo', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Sentiment Score')
    axes[0, 1].legend(['Sentiment Giornaliero', 'Media Mobile 7g'])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Correlazione Sentiment vs Variazione Prezzo
    if 'future_change_1d' in data.columns:
        valid_data = data.dropna(subset=['sentiment_mean', 'future_change_1d'])
        scatter = axes[0, 2].scatter(valid_data['sentiment_mean'], valid_data['future_change_1d'], 
                                   alpha=0.6, c=valid_data['sentiment_mean'], cmap='RdYlGn')
        axes[0, 2].set_title('Sentiment vs Variazione Prezzo 1g', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Sentiment Score')
        axes[0, 2].set_ylabel('Variazione Prezzo 1g (%)')
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 2])
    
    # 4. Distribuzione variazioni prezzo
    if 'price_change_1d' in data.columns:
        valid_changes = data['price_change_1d'].dropna()
        axes[1, 0].hist(valid_changes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Distribuzione Variazioni Prezzo 1g', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Variazione %')
        axes[1, 0].set_ylabel('Frequenza')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Volume tweet vs prezzo
    axes[1, 1].scatter(data['tweet_count'], data['price'], alpha=0.6, color='purple')
    axes[1, 1].set_title('Volume Tweet vs Prezzo', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Numero Tweet')
    axes[1, 1].set_ylabel('Prezzo USD')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Matrice correlazioni
    corr_cols = ['sentiment_mean', 'price_change_1d', 'tweet_count', 'price']
    available_corr_cols = [col for col in corr_cols if col in data.columns]
    
    if len(available_corr_cols) > 1:
        corr_data = data[available_corr_cols].corr()
        im = axes[1, 2].imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 2].set_xticks(range(len(available_corr_cols)))
        axes[1, 2].set_yticks(range(len(available_corr_cols)))
        axes[1, 2].set_xticklabels(available_corr_cols, rotation=45)
        axes[1, 2].set_yticklabels(available_corr_cols)
        axes[1, 2].set_title('Matrice Correlazioni', fontsize=14, fontweight='bold')
        
        # Aggiungi valori nella matrice
        for i in range(len(available_corr_cols)):
            for j in range(len(available_corr_cols)):
                axes[1, 2].text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                               ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()
    
    # Statistiche finali
    print("\n=== STATISTICHE FINALI ===")
    if 'sentiment_mean' in data.columns and 'future_change_1d' in data.columns:
        correlation = data['sentiment_mean'].corr(data['future_change_1d'])
        print(f"Correlazione sentiment-variazione prezzo 1g: {correlation:.4f}")
    
    print(f"Periodo analizzato: {data['date'].min()} - {data['date'].max()}")
    print(f"Giorni totali: {len(data)}")
    print(f"Tweet totali: {data['tweet_count'].sum()}")

def main():
    """Funzione principale con validazione temporale corretta"""
    print("=== ANALISI BITCOIN CON PREZZI REALI ===\n")
    
    # 1. Carica dataset sentiment
    print("1. Caricamento dataset...")
    df = load_and_clean_data("Dataset tesi.csv")
    if df is None:
        return
    
    # 2. Analisi sentiment
    print("\n2. Analisi sentiment...")
    df = perform_sentiment_analysis(df)
    
    # 3. Aggrega dati per giorno
    print("\n3. Aggregazione dati giornalieri...")
    daily_data = create_daily_features(df)
    
    # 4. Scarica prezzi Bitcoin reali
    print("\n4. Download prezzi Bitcoin reali...")
    start_date = daily_data['date'].min() - timedelta(days=10)
    end_date = daily_data['date'].max() + timedelta(days=10)
    
    real_prices = download_bitcoin_prices(start_date, end_date)
    if real_prices is None:
        print("Impossibile scaricare i prezzi. Verifica la connessione internet.")
        return
    
    # 5. Unisci dati
    print("\n5. Unione dati sentiment e prezzi...")
    merged_data = merge_with_real_prices(daily_data, real_prices)
    if merged_data is None:
        return
    
    # 6. Addestra modelli di previsione
    print("\n6. Training modelli di previsione...")
    
    # Modello 1 giorno
    model_1d, data_1d = train_prediction_model(merged_data, '1d')
    
    # Modello 3 giorni
    model_3d, data_3d = train_prediction_model(merged_data, '3d')
    
    # Modello 7 giorni  
    model_7d, data_7d = train_prediction_model(merged_data, '7d')
    
    # 7. Visualizzazioni
    print("\n7. Creazione visualizzazioni...")
    create_visualizations(merged_data)
    
    # 8. Salva risultati
    print("\n8. Salvataggio risultati...")
    try:
        merged_data.to_csv("risultati_analisi_completa.csv", index=False)
        print("✓ Risultati salvati in 'risultati_analisi_completa.csv'")
    except Exception as e:
        print(f"Errore salvataggio: {e}")
    
    print("\n=== ANALISI COMPLETATA ===")
    print("Il modello ora usa i prezzi REALI di Bitcoin per validare le previsioni!")

if __name__ == "__main__":
    main()