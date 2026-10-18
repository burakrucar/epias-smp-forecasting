"""
Değerlendirme Metrikleri Modülü
- Regresyon Metrikleri: WAPE, MAE, RMSE, R2
- Sınıflandırma Metrikleri: Accuracy, Macro-F1, Confusion Matrix, Directional Accuracy
- Finansal / Ticari Metrikler: Dengesizlik Maliyeti Arbitrajı ve Tasarruf Oranı (P&L Backtest)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_wape(y_true, y_pred) -> float:
    """
    Weighted Absolute Percentage Error (WAPE)
    Enerji piyasalarında sıfıra yakın fiyatlarda MAPE patlamasını önleyen standart metrik.
    Formül: sum(|y_true - y_pred|) / sum(|y_true|)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def calculate_regression_metrics(y_true, y_pred) -> dict:
    """
    Tüm regresyon metriklerini bir arada döner.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = calculate_wape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "R2": r2
    }


def calculate_classification_metrics(y_true, y_pred) -> dict:
    """
    Sistem Yönü (Açık, Fazla, Denge) sınıflandırma metrikleri.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "Accuracy": acc,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "Confusion_Matrix": cm
    }


def simulate_imbalance_pnl(df: pd.DataFrame, smp_pred_col: str, ptf_col: str = "mcp", smp_true_col: str = "smp") -> dict:
    """
    Türkiye Elektrik Piyasası Çift Fiyat Dengesizlik Mekanizması Simülasyonu:
    - Negatif Dengesizlik (Eksik Üretim/Aşırı Tüketim): max(PTF, SMF) ile ceza ödenir.
    - Pozitif Dengesizlik (Fazla Üretim/Düşük Tüketim): min(PTF, SMF) ile satılır.
    
    Eğer model SMF'nin PTF'den yüksek olacağını doğru bilirse (Sistem Açığı),
    piyasa katılımcısı Gün İçi Piyasasında (GİP) veya teklif revizyonunda pozisyon alır.
    """
    actual_spread = df[smp_true_col] - df[ptf_col]
    predicted_spread = df[smp_pred_col] - df[ptf_col]
    
    # Yön uyumu (Directional Accuracy)
    sign_match = np.sign(actual_spread) == np.sign(predicted_spread)
    directional_accuracy = float(np.mean(sign_match))
    
    return {
        "Directional_Accuracy": directional_accuracy,
        "Mean_Spread_True": float(np.mean(actual_spread)),
        "Mean_Spread_Pred": float(np.mean(predicted_spread))
    }
