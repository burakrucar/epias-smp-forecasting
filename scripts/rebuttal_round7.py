import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import sys
import os

sys.path.append("src")
from features import build_thesis_feature_matrix
from models import ThreeClassDirectionClassifier, PostDAMSpreadRegressor
from reproduce_all_thesis_results import calc_exact_pnl
from significance_tests import block_bootstrap_ci

print("=========================================================================================")
print("  T.C. İSTANBUL MEDENİYET ÜNİVERSİTESİ - TEZ DENEYLERİ ROUND 7 KAPSAMLI DENETİM")
print("=========================================================================================")

# 1. VERİ YÜKLEME VE BÖLÜMLEME
df_raw = pd.read_parquet("data/raw/epias_official_market_data_full.parquet")
df_f = df_raw.copy()
df_f["load_forecast_mw"] = df_f["load_forecast_mw"].ffill()
df_f["wind_forecast_mw"] = df_f["wind_forecast_mw"].ffill()
df_clean, cols_pre, cols_post = build_thesis_feature_matrix(df_f, lag_offset=14)

test_start = pd.Timestamp("2025-10-01 00:00:00+03:00")
test_end = pd.Timestamp("2026-09-30 23:00:00+03:00")
val_start = pd.Timestamp("2025-01-01 00:00:00+03:00")

tr_mask = df_clean["timestamp"] < val_start
val_mask = (df_clean["timestamp"] >= val_start) & (df_clean["timestamp"] < test_start)
trval_mask = df_clean["timestamp"] < test_start
test_mask = (df_clean["timestamp"] >= test_start) & (df_clean["timestamp"] <= test_end)

val_df = df_clean[val_mask].copy().reset_index(drop=True)
test_df = df_clean[test_mask].copy().reset_index(drop=True)

# ---------------------------------------------------------------------------------------
# [A] VALİDASYON KÜMESİ DENETİMİ VE IZGARA DUYARLILIĞI
# ---------------------------------------------------------------------------------------
print("\n--- [A] VALİDASYON KÜMESİ (2025 İLK 9 AY - N=6552) REFERANSLAR VE IZGARA DUYARLILIĞI ---")
val_ptf = val_df["ptf"].values
val_smf = val_df["smf"].values
val_neg = val_df["negativeImbalance"].values
val_pos = val_df["positiveImbalance"].values
val_ydir = val_df["target_direction"].values

val_mae_ptf = np.mean(np.abs(val_smf - val_ptf))
val_rmse_ptf = np.sqrt(np.mean((val_smf - val_ptf)**2))
val_pnl_maj = np.mean(calc_exact_pnl(np.ones(len(val_ydir)), val_ptf, val_neg, val_pos, val_ydir))
val_pnl_orc = np.mean(calc_exact_pnl(np.where(val_ydir==1, 1, np.where(val_ydir==-1, -1, 0)), val_ptf, val_neg, val_pos, val_ydir))

print(f"  SMF = PTF Naif Kuralı: MAE = {val_mae_ptf:6.2f} TL | RMSE = {val_rmse_ptf:6.2f} TL")
print(f"  Her Zaman Açık P&L:    {val_pnl_maj:6.2f} TL/MWh")
print(f"  Oracle P&L:            {val_pnl_orc:6.2f} TL/MWh")

# Grid search on validation set
print("\n  Aşama 2 Hiperparametre Izgara Araması (Train -> Predict Val):")
grid_results = []
for n_est in [100, 250, 400]:
    for lr in [0.03, 0.05]:
        for depth in [4, 5, 6]:
            m = lgb.LGBMRegressor(n_estimators=n_est, learning_rate=lr, max_depth=depth, min_child_samples=30, random_state=42, verbose=-1, n_jobs=-1)
            m.fit(df_clean.loc[tr_mask, cols_post], df_clean.loc[tr_mask, "target_smf"] - df_clean.loc[tr_mask, "ptf"])
            p_val = m.predict(df_clean.loc[val_mask, cols_post])
            y_val_sp = val_smf - val_ptf
            mae_g = np.mean(np.abs(p_val - y_val_sp))
            rmse_g = np.sqrt(np.mean((p_val - y_val_sp)**2))
            act_g = np.where(p_val > 0, 1, -1)
            pnl_g = np.mean(calc_exact_pnl(act_g, val_ptf, val_neg, val_pos, val_ydir))
            grid_results.append({
                "n_est": n_est, "lr": lr, "depth": depth,
                "MAE": mae_g, "RMSE": rmse_g, "PnL": pnl_g
            })

df_grid = pd.DataFrame(grid_results).sort_values("RMSE")
print(df_grid.head(6).to_string(index=False))

# ---------------------------------------------------------------------------------------
# [B] TEST KÜMESİ MODELLERİ VE EYLEMLER
# ---------------------------------------------------------------------------------------
ptf_test = test_df["ptf"].values
smf_test = test_df["smf"].values
neg_test = test_df["negativeImbalance"].values
pos_test = test_df["positiveImbalance"].values
y_dir_test = test_df["target_direction"].values
timestamps_test = test_df["timestamp"]
n_test = len(test_df)

# Canonical Stage 2
reg_spread = PostDAMSpreadRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=30, random_state=42)
reg_spread.fit(df_clean.loc[trval_mask, cols_post], df_clean.loc[trval_mask, "target_smf"] - df_clean.loc[trval_mask, "ptf"])
_, test_pred_sp = reg_spread.predict_smf(test_df[cols_post], ptf_test)
act_s2 = np.where(test_pred_sp > 0, 1, -1)
pay_s2 = calc_exact_pnl(act_s2, ptf_test, neg_test, pos_test, y_dir_test)

# Canonical Stage 1
clf_tr = ThreeClassDirectionClassifier()
clf_tr.fit(df_clean.loc[tr_mask, cols_pre], df_clean.loc[tr_mask, "target_direction"])
val_raw_prob = clf_tr.predict_proba(df_clean.loc[val_mask, cols_pre])
y_val = df_clean.loc[val_mask, "target_direction"].values
cal_def = LogisticRegression().fit(val_raw_prob[:, 2:3], (y_val == 1).astype(int))
cal_sur = LogisticRegression().fit(val_raw_prob[:, 0:1], (y_val == -1).astype(int))

clf_pre = ThreeClassDirectionClassifier()
clf_pre.fit(df_clean.loc[trval_mask, cols_pre], df_clean.loc[trval_mask, "target_direction"])
proba_dir_pre = clf_pre.predict_proba(test_df[cols_pre])

proba_cal = proba_dir_pre.copy()
proba_cal[:, 2] = cal_def.predict_proba(proba_dir_pre[:, 2:3])[:, 1]
proba_cal[:, 0] = cal_sur.predict_proba(proba_dir_pre[:, 0:1])[:, 1]
proba_cal[:, 1] = np.maximum(0.0, 1.0 - proba_cal[:, 2] - proba_cal[:, 0])

c_def = np.mean(df_clean.loc[trval_mask, "negativeImbalance"] - df_clean.loc[trval_mask, "ptf"])
c_sur = np.mean(df_clean.loc[trval_mask, "ptf"] - df_clean.loc[trval_mask, "positiveImbalance"])

ev_l_cal = proba_cal[:, 2] * c_def - (1.0 - proba_cal[:, 2]) * c_sur
ev_s_cal = proba_cal[:, 0] * c_sur - (1.0 - proba_cal[:, 0]) * c_def
act_s1_cal = np.zeros(n_test)
act_s1_cal[(ev_l_cal > 70.0) & (ev_l_cal > ev_s_cal)] = 1
act_s1_cal[(ev_s_cal > 70.0) & (ev_s_cal > ev_l_cal)] = -1
pay_s1_cal = calc_exact_pnl(act_s1_cal, ptf_test, neg_test, pos_test, y_dir_test)

act_maj = np.ones(n_test)
pay_maj = calc_exact_pnl(act_maj, ptf_test, neg_test, pos_test, y_dir_test)

# ---------------------------------------------------------------------------------------
# [C] GÜN BLOKLU PERMÜTASYON / KARIŞTIRMA TESTİ (YÖNSEL BECERİ VE GETİRİ YUMUŞATMA)
# ---------------------------------------------------------------------------------------
print("\n--- [B] GÜN BLOKLU PERMÜTASYON (BOŞ DAĞILIM) TESTİ (N = 2.000 Tekrar) ---")
n_days = 365
act_blocks = act_s1_cal.reshape(n_days, 24)
timestamps_days = timestamps_test.dt.strftime("%Y-%m").values

# Gerçek Aşama 1 değerleri
test_df["pay_s1_cal"] = pay_s1_cal
actual_pnl = np.mean(pay_s1_cal)
test_df["month"] = timestamps_days
monthly_s1 = test_df.groupby("month")[["pay_s1_cal"]].mean()
actual_monthly_sd = monthly_s1["pay_s1_cal"].std()

rng = np.random.default_rng(42)
perm_pnls = []
perm_sds = []

# Payoff function for vectorized day calculation
for _ in range(2000):
    shuffled_idx = rng.permutation(n_days)
    shuffled_acts = act_blocks[shuffled_idx].flatten()
    p_perm = calc_exact_pnl(shuffled_acts, ptf_test, neg_test, pos_test, y_dir_test)
    perm_pnls.append(np.mean(p_perm))
    
    # monthly SD of permuted
    test_df["temp_pay"] = p_perm
    m_sd = test_df.groupby("month")["temp_pay"].mean().std()
    perm_sds.append(m_sd)

p_val_pnl = np.mean(np.array(perm_pnls) >= actual_pnl)
p_val_sd = np.mean(np.array(perm_sds) <= actual_monthly_sd)

print(f"  Aşama 1 Gerçek P&L:             {actual_pnl:6.2f} TL/MWh")
print(f"  Boş Dağılım Ortalama P&L:       {np.mean(perm_pnls):6.2f} TL/MWh [%95 GA: {np.percentile(perm_pnls, 2.5):6.2f}, {np.percentile(perm_pnls, 97.5):6.2f}]")
print(f"  Yönsel Beceri P-Değeri:         p = {p_val_pnl:.4f} (Anlamlılık: {p_val_pnl < 0.05})")

print(f"\n  Aşama 1 Gerçek Aylık SD:        {actual_monthly_sd:6.2f} TL/MWh")
print(f"  Boş Dağılım Ortalama Aylık SD:  {np.mean(perm_sds):6.2f} TL/MWh [%95 GA: {np.percentile(perm_sds, 2.5):6.2f}, {np.percentile(perm_sds, 97.5):6.2f}]")
print(f"  Oynaklık Azaltma P-Değeri:      p = {p_val_sd:.4f}")

# Aktif Saat Başına P&L Yoğunluğu
act_hours_s1 = np.sum(act_s1_cal != 0)
pnl_per_active_s1 = np.sum(pay_s1_cal) / act_hours_s1
pnl_per_active_maj = np.sum(pay_maj) / n_test
print(f"\n  Aktif Saat Başına P&L Yoğunluğu:")
print(f"    Aşama 1 Kalibre: {pnl_per_active_s1:6.2f} TL/MWh (Aktif Saat: {act_hours_s1} / 8760 - %{act_hours_s1/n_test*100:.1f})")
print(f"    Her Zaman Açık:  {pnl_per_active_maj:6.2f} TL/MWh (Aktif Saat: 8760 / 8760 - %100.0)")

# ---------------------------------------------------------------------------------------
# [C] TABLO 4 DÜZELTMELERİ (AYLIK SHARPE, KÜMÜLATİF MDD VE AY SAYILARI)
# ---------------------------------------------------------------------------------------
print("\n--- [C] TABLO 4 DÜZELTMELERİ: AYLIK SHARPE, KÜMÜLATİF MDD (TL/MW) ---")
test_df["pay_maj"] = pay_maj
test_df["pay_s1_cal"] = pay_s1_cal
test_df["pay_s2"] = pay_s2

monthly_all = test_df.groupby("month")[["pay_maj", "pay_s1_cal", "pay_s2"]].mean()

# Negatif aylar kontrolü
neg_maj = (monthly_all["pay_maj"] < 0).sum()
neg_s1 = (monthly_all["pay_s1_cal"] < 0).sum()
print(f"  Negatif Getirili Ay Sayısı: Her Zaman Açık = {neg_maj} / 12, Aşama 1 = {neg_s1} / 12")

# Aylık Sharpe (N = 12 ay)
# Annualized Sharpe = sqrt(12) * mean / std
m_maj_mean, m_maj_sd = monthly_all["pay_maj"].mean(), monthly_all["pay_maj"].std()
m_s1_mean, m_s1_sd = monthly_all["pay_s1_cal"].mean(), monthly_all["pay_s1_cal"].std()
m_s2_mean, m_s2_sd = monthly_all["pay_s2"].mean(), monthly_all["pay_s2"].std()

sharpe_maj_m = (m_maj_mean / m_maj_sd) * np.sqrt(12)
sharpe_s1_m = (m_s1_mean / m_s1_sd) * np.sqrt(12)
sharpe_s2_m = (m_s2_mean / m_s2_sd) * np.sqrt(12)

print(f"  Aylıklandırılmış Sharpe Oranı (N=12 Ay):")
print(f"    Her Zaman Açık:  {sharpe_maj_m:6.2f}")
print(f"    Aşama 1 Kalibre: {sharpe_s1_m:6.2f}")
print(f"    Aşama 2 Yayılım: {sharpe_s2_m:6.2f}")

# Bootstrap CI for monthly SD and Sharpe (N=2000)
def boot_monthly_metrics(monthly_series):
    vals = monthly_series.values
    sd_boot = []
    sh_boot = []
    for _ in range(2000):
        b = rng.choice(vals, size=12, replace=True)
        m_b, s_b = np.mean(b), np.std(b, ddof=1)
        sd_boot.append(s_b)
        sh_boot.append((m_b / s_b) * np.sqrt(12) if s_b > 0 else 0)
    return (np.percentile(sd_boot, 2.5), np.percentile(sd_boot, 97.5)), (np.percentile(sh_boot, 2.5), np.percentile(sh_boot, 97.5))

sd_ci_maj, sh_ci_maj = boot_monthly_metrics(monthly_all["pay_maj"])
sd_ci_s1, sh_ci_s1 = boot_monthly_metrics(monthly_all["pay_s1_cal"])

print(f"  Aylık SD 95% GA: Her Zaman Açık = [{sd_ci_maj[0]:.2f}, {sd_ci_maj[1]:.2f}] | Aşama 1 = [{sd_ci_s1[0]:.2f}, {sd_ci_s1[1]:.2f}]")
print(f"  Aylık Sharpe 95% GA: Her Zaman Açık = [{sh_ci_maj[0]:.2f}, {sh_ci_maj[1]:.2f}] | Aşama 1 = [{sh_ci_s1[0]:.2f}, {sh_ci_s1[1]:.2f}]")

# Kümülatif Maksimum Çekilme (MDD) - Birim: TL/MW (1 MW Kapasite için)
# Her günün toplam net nakit akışı: sum(24 saatlik payoff * 1 MW)
daily_maj_tl = pay_maj.reshape(n_days, 24).sum(axis=1) # TL/MW/gün
daily_s1_tl = pay_s1_cal.reshape(n_days, 24).sum(axis=1) # TL/MW/gün
daily_s2_tl = pay_s2.reshape(n_days, 24).sum(axis=1)

def calc_cum_mdd_tl(daily_cashflows):
    cum = np.cumsum(daily_cashflows)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return np.max(dd)

mdd_maj_tl = calc_cum_mdd_tl(daily_maj_tl)
mdd_s1_tl = calc_cum_mdd_tl(daily_s1_tl)
mdd_s2_tl = calc_cum_mdd_tl(daily_s2_tl)

print(f"\n  Kümülatif Maksimum Çekilme (MDD - Birim: TL/MW):")
print(f"    Her Zaman Açık:  {mdd_maj_tl:10.2f} TL/MW")
print(f"    Aşama 1 Kalibre: {mdd_s1_tl:10.2f} TL/MW (%{(1 - mdd_s1_tl/mdd_maj_tl)*100:.1f} daha düşük çekilme)")
print(f"    Aşama 2 Yayılım: {mdd_s2_tl:10.2f} TL/MW")

# ---------------------------------------------------------------------------------------
# [D] DOĞAL HİZALANMA ORANI (ALPHA_WIND = %52.66) KESİN DEĞERLERİ VE GA
# ---------------------------------------------------------------------------------------
print("\n--- [D] EMPİRİK PORTFÖY VEKİLİ (ALPHA_WIND = %52.66) ANALİZİ ---")
alpha_w = 0.5266

def calc_general_alpha_pnl(actions, ptf, neg_p, pos_p, actual_dir, alpha):
    payoffs = np.zeros(len(actions))
    for i, a in enumerate(actions):
        if a == 1:
            if actual_dir[i] == 1:
                payoffs[i] = alpha * (neg_p[i] - ptf[i]) - (1.0 - alpha) * (ptf[i] - pos_p[i])
            else:
                payoffs[i] = -alpha * (ptf[i] - pos_p[i]) + (1.0 - alpha) * (neg_p[i] - ptf[i])
        elif a == -1:
            if actual_dir[i] == -1:
                payoffs[i] = alpha * (ptf[i] - pos_p[i]) - (1.0 - alpha) * (neg_p[i] - ptf[i])
            else:
                payoffs[i] = -alpha * (neg_p[i] - ptf[i]) + (1.0 - alpha) * (ptf[i] - pos_p[i])
        else:
            payoffs[i] = 0.0
    return payoffs

pay_s1_w = calc_general_alpha_pnl(act_s1_cal, ptf_test, neg_test, pos_test, y_dir_test, alpha_w)
pay_maj_w = calc_general_alpha_pnl(act_maj, ptf_test, neg_test, pos_test, y_dir_test, alpha_w)
pay_s2_w = calc_general_alpha_pnl(act_s2, ptf_test, neg_test, pos_test, y_dir_test, alpha_w)

_, s1_w_lb, s1_w_ub = block_bootstrap_ci(pay_s1_w, block_len=168, n_boot=2000, seed=42)
_, maj_w_lb, maj_w_ub = block_bootstrap_ci(pay_maj_w, block_len=168, n_boot=2000, seed=42)
_, s2_w_lb, s2_w_ub = block_bootstrap_ci(pay_s2_w, block_len=168, n_boot=2000, seed=42)

diff_w = pay_s1_w - pay_maj_w
_, diff_w_lb, diff_w_ub = block_bootstrap_ci(diff_w, block_len=168, n_boot=2000, seed=42)

print(f"  alpha_wind = %52.66 Altında P&L Sonuçları:")
print(f"    Her Zaman Açık:         {np.mean(pay_maj_w):6.2f} TL/MWh [%95 GA: {maj_w_lb:6.2f}, {maj_w_ub:6.2f}]")
print(f"    Aşama 1 Kalibre:        {np.mean(pay_s1_w):6.2f} TL/MWh [%95 GA: {s1_w_lb:6.2f}, {s1_w_ub:6.2f}]")
print(f"    Aşama 2 Yayılım:        {np.mean(pay_s2_w):6.2f} TL/MWh [%95 GA: {s2_w_lb:6.2f}, {s2_w_ub:6.2f}]")
print(f"    Fark [Aşama 1 - Açık]:  {np.mean(diff_w):+6.2f} TL/MWh [%95 GA: {diff_w_lb:6.2f}, {diff_w_ub:6.2f}]")

# ---------------------------------------------------------------------------------------
# [E] REJİM 3 SİMETRİK KIRPMA DOĞRU YÜZDELİK BOOTSTRAP GÜVEN ARALIĞI
# ---------------------------------------------------------------------------------------
print("\n--- [E] REJİM 3 SİMETRİK KIRPMA DOĞRU YÜZDELİK BOOTSTRAP GA ---")
r3 = timestamps_test >= pd.Timestamp("2026-04-04 00:00:00+03:00")
df_r3 = test_df[r3].copy()
spread_r3 = smf_test[r3] - ptf_test[r3]
q_low = np.percentile(spread_r3, 1.0)
q_high = np.percentile(spread_r3, 99.0)
trim_mask = (spread_r3 >= q_low) & (spread_r3 <= q_high)

diff_trim_hourly = pay_s1_cal[r3][trim_mask] - pay_maj[r3][trim_mask]
pt_est = np.mean(diff_trim_hourly)

# Moving block bootstrap percentile CI
n_trim = len(diff_trim_hourly)
block_len = 168
n_blocks = n_trim // block_len
rng = np.random.default_rng(42)
boot_diffs = []
for _ in range(2000):
    start_idx = rng.integers(0, n_trim - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in start_idx])
    boot_diffs.append(np.mean(diff_trim_hourly[idx]))

ci_low_trim = np.percentile(boot_diffs, 2.5)
ci_high_trim = np.percentile(boot_diffs, 97.5)
print(f"  Rejim 3 Simetrik Kırpılmış Saat Sayısı: {n_trim} / {r3.sum()} (Alt %1 ve Üst %1 elendi)")
print(f"  Nokta Tahmin (Aşama 1 Kalibre - Açık): {pt_est:+6.2f} TL/MWh")
print(f"  Doğru Yüzdelik Bootstrap %95 GA:       [{ci_low_trim:+6.2f}, {ci_high_trim:+6.2f}] TL/MWh")

# ---------------------------------------------------------------------------------------
# [F] POZİTİF DENGESİZLİK 1.538 SAATLİK KALINTI KIRILIMI
# ---------------------------------------------------------------------------------------
print("\n--- [F] 2026 YILI POZİTİF DENGESİZLİK 1.538 SAATLİK KALINTI DAĞILIMI ---")
p2_mask = test_df["timestamp"] >= pd.Timestamp("2026-01-01 00:00:00+03:00")
p2_df = test_df[p2_mask].copy()

# Karar 14030 statutory positive imbalance:
# Açık veya Dengede: min(ptf, smf) * 0.97
# Fazla: min(ptf, smf) * 0.94
pos_statutory = np.where(
    p2_df["systemStatus"] == "Enerji Fazlası",
    np.minimum(p2_df["ptf"], p2_df["smf"]) * 0.94,
    np.minimum(p2_df["ptf"], p2_df["smf"]) * 0.97
)

diff_pos = p2_df["positiveImbalance"] - pos_statutory
pos_mismatches = p2_df[np.abs(diff_pos) >= 0.1].copy()
pos_mismatches["residual"] = diff_pos[np.abs(diff_pos) >= 0.1]

zero_smf_mask = pos_mismatches["smf"] <= 0.01
ceiling_mask = (pos_mismatches["smf"] >= 3399.0) | (pos_mismatches["ptf"] >= 3399.0)
other_mask = (~zero_smf_mask) & (~ceiling_mask)

print(f"  Toplam Pozitif Uyuşmazlık: {len(pos_mismatches)} saat")
print(f"    1. Sıfır SMF Taban Bedeli (-106 TL): {zero_smf_mask.sum()} saat")
print(f"    2. Tavan Fiyat Saatleri:              {ceiling_mask.sum()} saat")
print(f"    3. Diğer Uzlaştırma Düzeltmeleri:     {other_mask.sum()} saat")

res_other = pos_mismatches.loc[other_mask, "residual"]
print(f"\n  Diğer {other_mask.sum()} Saatin Kalıntı İstatistikleri (Gerçek - Formül):")
print(f"    Ortalama Sapma: {res_other.mean():6.2f} TL")
print(f"    Standart Sapma: {res_other.std():6.2f} TL")
print(f"    Medyan Sapma:   {res_other.median():6.2f} TL")
print(f"    Min / Max:      {res_other.min():6.2f} TL / {res_other.max():6.2f} TL")
print(f"    Kuantiller (%25 / %75): {res_other.quantile(0.25):6.2f} TL / {res_other.quantile(0.75):6.2f} TL")
