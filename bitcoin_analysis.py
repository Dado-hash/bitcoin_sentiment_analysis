import pandas as pd 
from datetime import timedelta
import tweetnlp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
                    if len(df.columns) >= 3:  # Verifica che abbia almeno 3 colonne
                        print(f"Dataset caricato con encoding={encoding}, sep='{sep}'")
                        break
                except:
                    continue
            if df is not None:
                break
        
        if df is None:
            raise Exception("Impossibile caricare il dataset con i parametri testati")
        
        # Assegna nomi colonne se necessario
        if len(df.columns) >= 3:
            df.columns = ['timestamp', 'text', 'btc_price'] + list(df.columns[3:])
        else:
            raise Exception("Il dataset deve avere almeno 3 colonne")
        
        # Pulizia valori numerici del prezzo BTC
        if df['btc_price'].dtype == 'object':
            df['btc_price'] = df['btc_price'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        df['btc_price'] = pd.to_numeric(df['btc_price'], errors='coerce')
        
        # Conversione timestamp con multiple format
        timestamp_formats = ['%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M', '%Y/%m/%d %H:%M']
        df['timestamp'] = None
        
        for fmt in timestamp_formats:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt, errors='coerce')
                if df['timestamp'].notna().sum() > 0:
                    break
            except:
                continue
        
        # Se ancora non funziona, prova inference automatica
        if df['timestamp'].isna().all():
            df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        
        # Rimuovi righe con valori mancanti
        initial_len = len(df)
        df = df.dropna(subset=['timestamp', 'text', 'btc_price'])
        
        print(f"Dataset caricato: {len(df)} righe valide su {initial_len} totali")
        print(f"Range temporale: {df['timestamp'].min()} - {df['timestamp'].max()}")
        return df
        
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        return None

def perform_sentiment_analysis(df):
    """Esegue l'analisi del sentiment sui tweet"""
    print("Iniziando analisi del sentiment...")
    
    try:
        # Inizializza il modello di sentiment
        model = tweetnlp.load_model('sentiment') 
    except Exception as e:
        print(f"Errore nel caricamento del modello tweetnlp: {e}")
        print("Usando sentiment analysis semplificata basata su parole chiave...")
        return perform_simple_sentiment_analysis(df)
    
    # Analizza il sentiment per ogni tweet
    sentiments = []
    for i, text in enumerate(df['text']):
        try:
            # Converti a stringa e gestisci testi vuoti
            text_str = str(text).strip()
            if not text_str or text_str.lower() in ['nan', 'none', '']:
                sentiments.append('neutral')
                continue
                
            result = model.predict(text_str)
            sentiments.append(result['label'])
            
            if (i + 1) % 100 == 0:
                print(f"Processati {i + 1}/{len(df)} tweet")
                
        except Exception as e:
            print(f"Errore nell'analisi del tweet {i}: {e}")
            sentiments.append('neutral')  # Default fallback
    
    df['sentiment'] = sentiments
    
    # Mappatura a numeri: positive = 1, neutral = 0, negative = -1
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map).fillna(0)
    
    print("Analisi del sentiment completata!")
    print(f"Distribuzione sentiment: {df['sentiment'].value_counts().to_dict()}")
    return df

def perform_simple_sentiment_analysis(df):
    """Sentiment analysis semplificata basata su parole chiave"""
    positive_words = ['good', 'great', 'buy', 'bull', 'moon', 'rise', 'up', 'profit', 'gain']
    negative_words = ['bad', 'sell', 'bear', 'crash', 'down', 'loss', 'drop', 'fall']
    
    sentiments = []
    for text in df['text']:
        text_str = str(text).lower()
        pos_count = sum(1 for word in positive_words if word in text_str)
        neg_count = sum(1 for word in negative_words if word in text_str)
        
        if pos_count > neg_count:
            sentiments.append('positive')
        elif neg_count > pos_count:
            sentiments.append('negative')
        else:
            sentiments.append('neutral')
    
    df['sentiment'] = sentiments
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)
    
    return df

def create_prediction_features(df, horizon_days=10):
    """Crea le features per la previsione del prezzo futuro"""
    # Ordina per timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # CORREZIONE IMPORTANTE: Per evitare data leakage, usiamo solo dati passati
    # Invece di predire il prezzo futuro, prediciamo la direzione del prezzo
    df['price_change_1d'] = df['btc_price'].pct_change(periods=1)
    df['price_change_7d'] = df['btc_price'].pct_change(periods=7)
    
    # Media mobile del sentiment
    df['sentiment_ma_3'] = df['sentiment_score'].rolling(window=3, min_periods=1).mean()
    df['sentiment_ma_7'] = df['sentiment_score'].rolling(window=7, min_periods=1).mean()
    
    # Volatilità del prezzo
    df['price_volatility'] = df['btc_price'].rolling(window=7, min_periods=1).std()
    
    # Target: variazione del prezzo nei prossimi giorni (per validazione)
    # NOTA: Questo va usato solo per test, non per training in produzione
    df['future_price_change'] = df['btc_price'].pct_change(periods=-horizon_days)
    df['future_direction'] = (df['future_price_change'] > 0).astype(int)
    
    # Rimuovi righe con valori mancanti per il training
    df_clean = df.dropna(subset=['sentiment_score', 'price_change_1d', 'future_price_change'])
    
    print(f"Features create per {len(df_clean)} osservazioni")
    return df_clean

def train_regression_model(df):
    """Addestra il modello di regressione lineare"""
    # Caratteristiche: usa solo dati passati
    feature_cols = ['sentiment_score', 'sentiment_ma_3', 'sentiment_ma_7', 
                   'price_change_1d', 'price_change_7d', 'price_volatility']
    
    # Rimuovi feature con tutti NaN
    available_features = [col for col in feature_cols if col in df.columns and not df[col].isna().all()]
    
    X = df[available_features]
    y = df['future_price_change']  # Target: variazione percentuale futura
    
    # Rimuovi outlier estremi
    q1, q3 = y.quantile([0.05, 0.95])
    mask = (y >= q1) & (y <= q3)
    X, y = X[mask], y[mask]
    
    if len(X) < 10:
        print("Dati insufficienti per il training")
        return None, df
    
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
    print(f"MSE Training: {train_mse:.6f}")
    print(f"MSE Test: {test_mse:.6f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nImportanza delle features:")
    print(feature_importance)
    
    return model, df

def create_visualizations(df):
    """Crea visualizzazioni dei risultati"""
    # Usa uno stile matplotlib valido
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figura con subplot multipli
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Prezzo BTC nel tempo
    axes[0, 0].plot(df['timestamp'], df['btc_price'], label='Prezzo BTC', alpha=0.8, linewidth=1.5)
    axes[0, 0].set_xlabel("Data")
    axes[0, 0].set_ylabel("Prezzo BTC (€)")
    axes[0, 0].set_title("Evoluzione Prezzo Bitcoin")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribuzione del sentiment
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        axes[0, 1].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                      autopct='%1.1f%%', colors=colors)
        axes[0, 1].set_title("Distribuzione del Sentiment")
    
    # 3. Sentiment vs Variazione Prezzo
    if 'sentiment_score' in df.columns and 'price_change_1d' in df.columns:
        scatter = axes[1, 0].scatter(df['sentiment_score'], df['price_change_1d'], 
                                   alpha=0.6, c=df['sentiment_score'], cmap='RdYlGn')
        axes[1, 0].set_xlabel("Sentiment Score")
        axes[1, 0].set_ylabel("Variazione Prezzo 1g (%)")
        axes[1, 0].set_title("Sentiment vs Variazione Prezzo")
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0])
    
    # 4. Correlazione tra variabili
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        corr_cols = ['sentiment_score', 'price_change_1d', 'price_change_7d']
        available_corr_cols = [col for col in corr_cols if col in df.columns]
        
        if len(available_corr_cols) > 1:
            corr_matrix = df[available_corr_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes[1, 1], square=True)
            axes[1, 1].set_title("Matrice di Correlazione")
    
    plt.tight_layout()
    plt.show()
    
    # Analisi correlazioni
    if 'sentiment_score' in df.columns and 'future_direction' in df.columns:
        correlation = df['sentiment_score'].corr(df['future_direction'])
        print(f"\nCorrelazione sentiment-direzione prezzo: {correlation:.4f}")
    
    # Statistiche descrittive
    print("\nStatistiche descrittive:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())

def main():
    """Funzione principale"""
    print("=== Analisi Sentiment e Previsione Prezzo Bitcoin ===\n")
    
    # 1. Carica e pulisce i dati
    df = load_and_clean_data("Dataset tesi.csv")
    if df is None:
        print("Impossibile caricare il dataset. Controlla il file e riprova.")
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
    try:
        output_file = "risultati_analisi.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nRisultati salvati in '{output_file}'")
    except Exception as e:
        print(f"Errore nel salvataggio: {e}")
    
    print("\n=== Analisi completata ===")

if __name__ == "__main__":
    main()