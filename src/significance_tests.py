"""
Gelişmiş İstatistiksel Doğrulama ve Ekonometrik Hipotez Testleri Modülü
- Günlük Agregasyonlu Diebold-Mariano Testi (273 Bağımsız Gün - MAE ve MSE Kayıpları)
- Eşleştirilmiş Günlük Wilcoxon Signed-Rank Testi (Sınıflandırma Doğruluğu Karşılaştırması)
- Sınıf Bazında Ayrıntılı Metrikler (Precision, Recall, F1 - Enerji Açığı, Fazlası, Dengede)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def daily_diebold_mariano_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray, power: int = 1, max_lags: int = 7) -> dict:
    """
    Saatlik otokorelasyondan arındırılmış GÜNLÜK Diebold-Mariano Testi.
    Günün 24 saatinin kayıpları toplanarak her gün için tek bir kayıp elde edilir (N = 273 gün).
    
    power = 1: Mutlak Hata (MAE)
    power = 2: Karesel Hata (MSE)
    """
    n_hours = len(y_true)
    n_days = n_hours // 24
    
    # 24 saatlik bloklara böl
    y_true_daily = y_true[:n_days * 24].reshape(n_days, 24)
    y_p1_daily = y_pred1[:n_days * 24].reshape(n_days, 24)
    y_p2_daily = y_pred2[:n_days * 24].reshape(n_days, 24)

    if power == 1:
        loss1 = np.mean(np.abs(y_true_daily - y_p1_daily), axis=1)
        loss2 = np.mean(np.abs(y_true_daily - y_p2_daily), axis=1)
    elif power == 2:
        loss1 = np.mean((y_true_daily - y_p1_daily) ** 2, axis=1)
        loss2 = np.mean((y_true_daily - y_p2_daily) ** 2, axis=1)
    else:
        raise ValueError("power 1 veya 2 olmalı")

    # Günlük kayıp farkı serisi: d_t = Loss(Model 1) - Loss(Model 2)
    # d_t > 0 ise Model 2 daha başarılıdır.
    d = loss1 - loss2
    n = len(d)
    mean_d = np.mean(d)

    # Newey-West HAC Otokorelasyon Düzeltmesi (Günlük düzeyde max_lags = 7 gün)
    gamma0 = np.var(d, ddof=1)
    gammas = []
    for lag in range(1, max_lags + 1):
        if lag < n:
            weight = 1.0 - (lag / (max_lags + 1.0))  # Bartlett çekirdeği
            cov = np.cov(d[:-lag], d[lag:])[0, 1]
            gammas.append(weight * cov)

    var_d = (gamma0 + 2.0 * np.sum(gammas)) / n
    if var_d <= 0 or np.isnan(var_d):
        dm_stat = 0.0
        p_val = 1.0
    else:
        dm_stat = mean_d / np.sqrt(var_d)
        p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "N_Days": n,
        "DM_Stat": dm_stat,
        "P_Value": p_val,
        "Significant_at_05": p_val < 0.05,
        "Mean_Daily_Loss_Diff": mean_d,
        "Power": "MAE" if power == 1 else "MSE"
    }


def daily_classification_wilcoxon_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> dict:
    """
    Saatlik bağımlılıktan arındırılmış GÜNLÜK Sınıflandırma Doğruluk Karşılaştırması.
    Her günün doğruluk oranı hesaplanır (N = 273 gün) ve Wilcoxon Signed-Rank Testi uygulanır.
    """
    n_hours = len(y_true)
    n_days = n_hours // 24
    
    y_t_d = y_true[:n_days * 24].reshape(n_days, 24)
    p1_d = y_pred1[:n_days * 24].reshape(n_days, 24)
    p2_d = y_pred2[:n_days * 24].reshape(n_days, 24)

    acc1_daily = np.mean(y_t_d == p1_d, axis=1)
    acc2_daily = np.mean(y_t_d == p2_d, axis=1)

    diff = acc2_daily - acc1_daily
    # Sıfır farkları filtrele
    non_zero = diff[diff != 0]
    if len(non_zero) < 10:
        stat, p_val = 0.0, 1.0
    else:
        stat, p_val = stats.wilcoxon(acc2_daily, acc1_daily)

    return {
        "N_Days": n_days,
        "Mean_Acc1": float(np.mean(acc1_daily)),
        "Mean_Acc2": float(np.mean(acc2_daily)),
        "Wilcoxon_Stat": float(stat),
        "P_Value": float(p_val),
        "Significant_at_05": p_val < 0.05
    }


def get_detailed_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_labels=[-1, 0, 1], class_names=["Enerji Fazlası", "Dengede", "Enerji Açığı"]) -> pd.DataFrame:
    """
    Sınıf bazında Precision, Recall, F1-skor tablosu üretir.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, zero_division=0
    )
    
    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            "Sınıf": name,
            "Precision": precision[i],
            "Recall": recall[i],
            "F1-Score": f1[i],
            "Destek (Gözlem)": support[i]
        })
    return pd.DataFrame(rows)


def calculate_brier_skill_score(y_true: np.ndarray, probas: np.ndarray, train_priors: np.ndarray) -> dict:
    """
    3 Sınıflı Brier Skoru ve Klimatolojik Referansa Göre Brier Beceri Skoru (BSS).
    BSS = 1 - (BS_model / BS_clim)
    """
    n = len(y_true)
    y_onehot = np.zeros((n, 3))
    # Haritalama: -1 -> 0, 0 -> 1, 1 -> 2
    for i, val in enumerate(y_true):
        if val == -1: y_onehot[i, 0] = 1.0
        elif val == 0: y_onehot[i, 1] = 1.0
        elif val == 1: y_onehot[i, 2] = 1.0

    bs_model = float(np.mean(np.sum((probas - y_onehot) ** 2, axis=1)))
    
    # Klimatoloji: Eğitim setindeki ampirik sınıf oranları
    clim_probas = np.tile(train_priors, (n, 1))
    bs_clim = float(np.mean(np.sum((clim_probas - y_onehot) ** 2, axis=1)))

    bss = 1.0 - (bs_model / bs_clim) if bs_clim > 0 else 0.0

    return {
        "BS_Model": bs_model,
        "BS_Climatology": bs_clim,
        "BSS": bss,
        "BSS_Percentage": bss * 100.0
    }


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """
    Çoklu hipotez testleri için Holm-Bonferroni (step-down) düzeltmesi.
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    results = [None] * m

    reject = True
    for rank, idx in enumerate(sorted_indices):
        p_val = p_values[idx]
        adj_alpha = alpha / (m - rank)
        if reject and p_val <= adj_alpha:
            is_sig = True
        else:
            is_sig = False
            reject = False  # Step-down kuralı gereği sonraki testler reddedilemez
        
        results[idx] = {
            "Original_P": p_val,
            "Rank": rank + 1,
            "Threshold": adj_alpha,
            "Significant_Adjusted": is_sig
        }
    return results


def block_bootstrap_ci(series: np.ndarray, block_len: int = 7, n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """
    Otokorelasyonlu günlük seriler için Hareketli Blok Bootstrap (Moving Block Bootstrap).
    Döndürür: (point_mean, ci_lower, ci_upper)
    """
    n = len(series)
    n_blocks = n // block_len
    rng = np.random.default_rng(seed)
    boot_means = []

    for _ in range(n_boot):
        start_indices = rng.integers(0, n - block_len + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in start_indices])
        boot_means.append(np.mean(series[idx]))

    alpha_low = (1.0 - ci) / 2.0 * 100.0
    alpha_high = (1.0 + ci) / 2.0 * 100.0
    low = float(np.percentile(boot_means, alpha_low))
    high = float(np.percentile(boot_means, alpha_high))
    return float(np.mean(series)), low, high

