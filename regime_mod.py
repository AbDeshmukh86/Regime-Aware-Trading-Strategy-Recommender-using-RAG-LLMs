import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import joblib
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

def generate_regime_analysis_report(
    symbol, 
    model_path="regime_model.joblib", 
    history_length=15
):
    """
    Fetches weekly data for a user-specified asset, computes technical features, 
    applies a pre-trained HMM model, and returns a Python dictionary regime analysis report.
    
    Parameters:
    -----------
    symbol : str
        The ticker symbol for yfinance (e.g., "^NSEI", "AAPL", "GC=F").
    model_path : str, optional
        Path to the pre-trained joblib model bundle (default is "regime_model.joblib").
    history_length : int, optional
        Number of historical weeks to include in the regime history (default is 15).
        
    Returns:
    --------
    dict
        A Python dictionary containing the asset regime analysis.
    """
    warnings.filterwarnings('ignore')
    
    # -----------------------------
    # 1. Download Weekly OHLCV
    # -----------------------------
    df = yf.download(symbol, period="5y", interval="1wk", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    
    if df.empty:
        return {"error": f"No data found for symbol: {symbol}"}
    
    # -----------------------------
    # 2. Feature Engineering
    # -----------------------------
    # EMA & Distance
    df["EMA40"] = df["Close"].ewm(span=40, adjust=False).mean()
    df["EMA_Distance"] = ((df["Close"] - df["EMA40"]) / df["EMA40"]) * 100
    
    # HH / HL Score
    df["HH"] = (df["High"] > df["High"].shift(1)).astype(int)
    df["HL"] = (df["Low"] > df["Low"].shift(1)).astype(int)
    df["LH"] = (df["High"] < df["High"].shift(1)).astype(int)
    df["LL"] = (df["Low"] < df["Low"].shift(1)).astype(int)
    df["HH_HL_Score"] = df["HH"] + df["HL"] - df["LH"] - df["LL"]
    
    # ADX & Streaks
    adx = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ADX"] = adx.adx()
    df["ADX_Slope"] = df["ADX"].diff(3) / 3
    
    adx_streak = [0]
    for i in range(1, len(df)):
        if df["ADX"].iloc[i] > df["ADX"].iloc[i-1]:
            adx_streak.append(adx_streak[-1] + 1 if adx_streak[-1] > 0 else 1)
        elif df["ADX"].iloc[i] < df["ADX"].iloc[i-1]:
            adx_streak.append(adx_streak[-1] - 1 if adx_streak[-1] < 0 else -1)
        else:
            adx_streak.append(0)
    df["ADX_Streak"] = adx_streak
    
    # ATR & Range
    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ATR"] = (atr.average_true_range() / df["Close"]) * 100
    df["ATR_Slope"] = df["ATR"].diff(3) / 3
    df["Weekly_Range"] = ((df["High"] - df["Low"]) / df["Close"]) * 100
    df["Range_Relative"] = df["Weekly_Range"] / df["Weekly_Range"].shift(1)
    
    # Bollinger Bands
    rolling_std = df["EMA40"].rolling(10).std()
    df["BB_Width"] = (((df["EMA40"] + 1.5 * rolling_std) - (df["EMA40"] - 1.5 * rolling_std)) / df["EMA40"]) * 100
    
    # ROC
    df["ROC_1"] = df["Close"].pct_change(1) * 100
    df["ROC_4"] = df["Close"].pct_change(4) * 100
    
    # Candle Streak
    c_streak = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Open"].iloc[i]:
            c_streak.append(c_streak[-1] + 1 if c_streak[-1] > 0 else 1)
        elif df["Close"].iloc[i] < df["Open"].iloc[i]:
            c_streak.append(c_streak[-1] - 1 if c_streak[-1] < 0 else -1)
        else:
            c_streak.append(0)
    df["Candle_Streak"] = c_streak
    
    # Efficiency & Volume
    df["Efficiency_Ratio"] = (df["Close"] - df["Close"].shift(5)).abs() / df["Close"].diff().abs().rolling(5).sum()
    df["Relative_Volume"] = df["Volume"] / df["Volume"].rolling(4).mean()
    
    # Rolling R2
    r2 = [np.nan] * len(df)
    for i in range(4, len(df)):
        y = df["Close"].iloc[i-4:i+1].values
        x = np.arange(5)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2[i] = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    df["Rolling_R2"] = r2

    # -----------------------------
    # 3. Load Model & Preprocess
    # -----------------------------
    bundle = joblib.load(model_path)
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    hmm = bundle["hmm_final"]
    regime_labels = bundle["regime_labels"]
    feature_columns = bundle["feature_columns"]
    
    # Clean data & isolate features
    main_df = df.dropna(subset=feature_columns).copy()
    X = main_df[feature_columns].copy()
    
    # Scale and PCA
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # -----------------------------
    # 4. HMM Inference
    # -----------------------------
    log_prob, posterior = hmm.score_samples(X_pca)
    main_df["State"] = posterior.argmax(axis=1)
    main_df["Confidence"] = posterior.max(axis=1)
    main_df["Regime"] = main_df["State"].map(regime_labels)

    # -----------------------------
    # 5. Format Dictionary Output
    # -----------------------------
    latest_date = main_df.index[-1]
    latest_row = main_df.iloc[-1]
    latest_probs = posterior[-1]
    
    # Map probabilities to labels
    prob_dict = {regime_labels[i]: float(latest_probs[i]) for i in range(len(regime_labels))}
    
    # Extract historical timeline
    regime_history = []
    for date, row in main_df.tail(history_length).iterrows():
        regime_history.append({
            "date": date.strftime("%Y-%m-%d"),
            "regime": row["Regime"],
            "confidence": float(row["Confidence"])
        })
        
    output = {
        "asset": {
            "symbol": symbol,
        },
        "analysis_date": latest_date.strftime("%Y-%m-%d"),
        "model_metadata": {
            "features_count": len(feature_columns),
            "pca_components": pca.n_components_ if hasattr(pca, 'n_components_') else 9,
            "hmm_components": hmm.n_components if hasattr(hmm, 'n_components') else len(regime_labels),
            "regimes_mapped": regime_labels
        },
        "current_regime": {
            "state": int(latest_row["State"]),
            "name": latest_row["Regime"],
            "confidence": float(latest_row["Confidence"])
        },
        "probabilities": prob_dict,
        "raw_probabilities": latest_probs.tolist(),
        "latest_ohlcv": {
            "Open": float(latest_row["Open"]),
            "High": float(latest_row["High"]),
            "Low": float(latest_row["Low"]),
            "Close": float(latest_row["Close"]),
            "Volume": float(latest_row["Volume"])
        },
        "latest_features": {feat: float(latest_row[feat]) for feat in feature_columns},
        "regime_history": regime_history
    }
    
    return output