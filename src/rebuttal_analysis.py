import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import sys
import os

sys.path.append("src")
from features import build_thesis_feature_matrix
from significance_tests import block_bootstrap_ci

def calc_exact_pnl(actions, ptf, neg_p, pos_p, actual_dir):
    payoffs = np.zeros(len(actions))
    for i, a in enumerate(actions):
        if a == 1:
            if actual_dir[i] == 1:
                payoffs[i] = neg_p[i] - ptf[i]
            else:
                payoffs[i] = -(ptf[i] - pos_p[i])
        elif a == -1:
            if actual_dir[i] == -1:
                payoffs[i] = ptf[i] - pos_p[i]
            else:
                payoffs[i] = -(neg_p[i] - ptf[i])
        else:
            payoffs[i] = 0.0
    return payoffs

print("--- 1. VERİ YÜKLEME VE ÖZELLİK ÜRETİMİ ---")
df_raw = pd.read_parquet("data/raw/epias_official_market_data_full.parquet")
df_f = df_raw.copy()
df_f["load_forecast_mw"] = df_f["load_forecast_mw"].ffill()
df_f["wind_forecast_mw"] = df_f["wind_forecast_mw"].ffill()
df_clean, cols_gop, cols_dam = build_thesis_feature_matrix(df_f, lag_offset=14)

test_start = pd.Timestamp("2025-10-01 00:00:00+03:00")
test_end = pd.Timestamp("2026-09-30 23:00:00+03:00")
val_start = pd.Timestamp("2025-01-01 00:00:00+03:00")

tr_mask = df_clean["timestamp"] < val_start
val_mask = (df_clean["timestamp"] >= val_start) & (df_clean["timestamp"] < test_start)
tr_val_mask = df_clean["timestamp"] < test_start
test_mask = (df_clean["timestamp"] >= test_start) & (df_clean["timestamp"] <= test_end)

df_test = df_clean[test_mask].copy().reset_index(drop=True)
ptf_test = df_test["ptf"].values
smf_test = df_test["smf"].values
neg_test = df_test["negativeImbalance"].values
pos_test = df_test["positiveImbalance"].values
y_dir_test = df_test["target_direction"].values
timestamps_test = df_test["timestamp"]

# 2. MODELLERİN EĞİTİLMESİ
y_dir = df_clean["target_direction"] + 1 # 0: Fazla, 1: Denge, 2: Açık
clf_params = dict(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1, objective="multiclass", num_class=3)

# Train on Train (2024) -> Predict Val (2025 İlk 9 Ay) for Platt
clf_tr = lgb.LGBMClassifier(**clf_params).fit(df_clean.loc[tr_mask, cols_gop], y_dir[tr_mask])
val_probs = clf_tr.predict_proba(df_clean.loc[val_mask, cols_gop])
val_true = y_dir[val_mask]

# Platt Calibrator
calibrator = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
calibrator.fit(val_probs, val_true)

# Final classifier on Train+Val
clf_final = lgb.LGBMClassifier(**clf_params).fit(df_clean.loc[tr_val_mask, cols_gop], y_dir[tr_val_mask])
test_probs_raw = clf_final.predict_proba(df_clean.loc[test_mask, cols_gop])
test_probs_cal = calibrator.predict_proba(test_probs_raw)

# Out-of-fold check on Train+Val (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_tv = df_clean.loc[tr_val_mask, cols_gop].values
y_tv = y_dir[tr_val_mask].values
oof_probs = np.zeros((tr_val_mask.sum(), 3))
for tr_idx, v_idx in kf.split(X_tv):
    f_clf = lgb.LGBMClassifier(**clf_params).fit(X_tv[tr_idx], y_tv[tr_idx])
    oof_probs[v_idx] = f_clf.predict_proba(X_tv[v_idx])

calibrator_oof = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
calibrator_oof.fit(oof_probs, y_tv)
test_probs_cal_oof = calibrator_oof.predict_proba(test_probs_raw)

print(f"Val-Calibrated vs OOF-Calibrated Probs Correlation (Açık): {np.corrcoef(test_probs_cal[:,2], test_probs_cal_oof[:,2])[0,1]:.4f}")

# Aşama 1 Karar Kuralları
val_df = df_clean[val_mask]
mean_c_def = float(np.mean(val_df.loc[val_df["target_direction"] == 1, "negativeImbalance"] - val_df.loc[val_df["target_direction"] == 1, "ptf"]))
mean_c_sur = float(np.mean(val_df.loc[val_df["target_direction"] == -1, "ptf"] - val_df.loc[val_df["target_direction"] == -1, "positiveImbalance"]))

def get_stage1_actions(probs, tau=70.0):
    p_surplus = probs[:, 0]
    p_deficit = probs[:, 2]
    e_long = p_deficit * mean_c_def - (1.0 - p_deficit) * mean_c_sur
    e_short = p_surplus * mean_c_sur - (1.0 - p_surplus) * mean_c_def
    actions = np.zeros(len(probs))
    for i in range(len(probs)):
        if e_long[i] > tau and e_long[i] > e_short[i]:
            actions[i] = 1
        elif e_short[i] > tau and e_short[i] > e_long[i]:
            actions[i] = -1
        else:
            actions[i] = 0
    return actions

actions_st1_raw = get_stage1_actions(test_probs_raw, tau=70.0)
actions_st1_cal = get_stage1_actions(test_probs_cal, tau=70.0)
actions_st1_cal_oof = get_stage1_actions(test_probs_cal_oof, tau=70.0)

print("\nAşama 1 Eylem Dağılımı:")
print(f"  Ham Olasılıklar:     Long={np.sum(actions_st1_raw==1)} (%{np.mean(actions_st1_raw==1)*100:.1f}), Short={np.sum(actions_st1_raw==-1)} (%{np.mean(actions_st1_raw==-1)*100:.1f}), Pasif={np.sum(actions_st1_raw==0)} (%{np.mean(actions_st1_raw==0)*100:.1f})")
print(f"  Platt Kalibre:       Long={np.sum(actions_st1_cal==1)} (%{np.mean(actions_st1_cal==1)*100:.1f}), Short={np.sum(actions_st1_cal==-1)} (%{np.mean(actions_st1_cal==-1)*100:.1f}), Pasif={np.sum(actions_st1_cal==0)} (%{np.mean(actions_st1_cal==0)*100:.1f})")
print(f"  OOF Kalibre:         Long={np.sum(actions_st1_cal_oof==1)} (%{np.mean(actions_st1_cal_oof==1)*100:.1f}), Short={np.sum(actions_st1_cal_oof==-1)} (%{np.mean(actions_st1_cal_oof==1)*100:.1f}), Pasif={np.sum(actions_st1_cal_oof==0)} (%{np.mean(actions_st1_cal_oof==0)*100:.1f})")

# Stage 2 Model (Spread Regressor)
y_spread = df_clean["target_smf"] - df_clean["ptf"]
sp_reg = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
sp_reg.fit(df_clean.loc[tr_val_mask, cols_dam], y_spread[tr_val_mask])
pred_spread_test = sp_reg.predict(df_clean.loc[test_mask, cols_dam])
actions_st2 = np.where(pred_spread_test > 0, 1, -1)

# Actions for baselines
actions_maj = np.ones(len(df_test))
actions_orc = np.where(y_dir_test == 1, 1, np.where(y_dir_test == -1, -1, 0))

# Payoff vectors
pay_st1_raw = calc_exact_pnl(actions_st1_raw, ptf_test, neg_test, pos_test, y_dir_test)
pay_st1_cal = calc_exact_pnl(actions_st1_cal, ptf_test, neg_test, pos_test, y_dir_test)
pay_st1_cal_oof = calc_exact_pnl(actions_st1_cal_oof, ptf_test, neg_test, pos_test, y_dir_test)
pay_st2 = calc_exact_pnl(actions_st2, ptf_test, neg_test, pos_test, y_dir_test)
pay_maj = calc_exact_pnl(actions_maj, ptf_test, neg_test, pos_test, y_dir_test)
pay_orc = calc_exact_pnl(actions_orc, ptf_test, neg_test, pos_test, y_dir_test)

print("\n--- 3. TAM TEST YILI GENEL P&L VE GÜVEN ARALIKLARI ---")
for name, p_vec in [
    ("Oracle", pay_orc),
    ("Aşama 1 (Ham)", pay_st1_raw),
    ("Aşama 1 (Platt Kalibre)", pay_st1_cal),
    ("Aşama 1 (OOF Kalibre)", pay_st1_cal_oof),
    ("Aşama 2 (Yayılım)", pay_st2),
    ("Her Zaman Açık", pay_maj),
]:
    _, lb, ub = block_bootstrap_ci(p_vec, block_len=168, n_boot=2000, seed=42)
    print(f"  {name:25s}: {np.mean(p_vec):6.2f} TL/MWh  [%95 GA: {lb:6.2f}, {ub:6.2f}]")

# 4. REJİM BAZINDA AYRIŞTIRMA (DOĞRU EPDK KESİM TARİHLERİ İLE)
r1_mask = timestamps_test < pd.Timestamp("2026-01-01 00:00:00+03:00")
r2_mask = (timestamps_test >= pd.Timestamp("2026-01-01 00:00:00+03:00")) & (timestamps_test < pd.Timestamp("2026-04-04 00:00:00+03:00"))
r3_mask = timestamps_test >= pd.Timestamp("2026-04-04 00:00:00+03:00")

regimes = [
    ("Rejim 1 (2025 Q4: Eski DUY, Tavan=3400)", r1_mask),
    ("Rejim 2 (2026 Q1: Karar 14030, Tavan=3400)", r2_mask),
    ("Rejim 3 (2026 Q2-Q3: Karar 14030, Tavan=4500)", r3_mask),
]

print("\n--- 4. DETAYLI REJİM BAZINDA AYRIŞTIRMA (AŞAMA 1, AŞAMA 2, GÜVEN ARALIKLARI VE ORACLE ORANI) ---")
for r_name, r_m in regimes:
    n_hours = r_m.sum()
    orc_m = np.mean(pay_orc[r_m])
    maj_m = np.mean(pay_maj[r_m])
    st1_raw_m = np.mean(pay_st1_raw[r_m])
    st1_cal_m = np.mean(pay_st1_cal[r_m])
    st2_m = np.mean(pay_st2[r_m])
    
    _, maj_lb, maj_ub = block_bootstrap_ci(pay_maj[r_m], block_len=168, n_boot=2000, seed=42)
    _, st1_cal_lb, st1_cal_ub = block_bootstrap_ci(pay_st1_cal[r_m], block_len=168, n_boot=2000, seed=42)
    _, st2_lb, st2_ub = block_bootstrap_ci(pay_st2[r_m], block_len=168, n_boot=2000, seed=42)
    
    diff_st1 = pay_st1_cal[r_m] - pay_maj[r_m]
    _, d1_lb, d1_ub = block_bootstrap_ci(diff_st1, block_len=168, n_boot=2000, seed=42)
    
    diff_st2 = pay_st2[r_m] - pay_maj[r_m]
    _, d2_lb, d2_ub = block_bootstrap_ci(diff_st2, block_len=168, n_boot=2000, seed=42)
    
    print(f"\n{r_name} (N = {n_hours} saat):")
    print(f"  Oracle Üst Sınır:            {orc_m:6.2f} TL/MWh (100.0%)")
    print(f"  Her Zaman Açık:             {maj_m:6.2f} TL/MWh [%95 GA: {maj_lb:6.2f}, {maj_ub:6.2f}] (Oracle payı: %{maj_m/orc_m*100:4.1f})")
    print(f"  Aşama 1 (Ham Olasılık):     {st1_raw_m:6.2f} TL/MWh (Oracle payı: %{st1_raw_m/orc_m*100:4.1f})")
    print(f"  Aşama 1 (Platt Kalibre):    {st1_cal_m:6.2f} TL/MWh [%95 GA: {st1_cal_lb:6.2f}, {st1_cal_ub:6.2f}] (Oracle payı: %{st1_cal_m/orc_m*100:4.1f})")
    print(f"  Aşama 2 (Yayılım Modeli):   {st2_m:6.2f} TL/MWh [%95 GA: {st2_lb:6.2f}, {st2_ub:6.2f}] (Oracle payı: %{st2_m/orc_m*100:4.1f})")
    print(f"  Fark [Aşama 1 Kalibre - Açık]: {np.mean(diff_st1):+6.2f} TL/MWh [%95 GA: {d1_lb:6.2f}, {d1_ub:6.2f}]")
    print(f"  Fark [Aşama 2 Yayılım - Açık]: {np.mean(diff_st2):+6.2f} TL/MWh [%95 GA: {d2_lb:6.2f}, {d2_ub:6.2f}]")

# 5. REJİM 3 AY-AY AYRIŞTIRMASI VE AŞIRI SAATLER
print("\n--- 5. REJİM 3 AYLIK P&L VE EKSTREM SAATLERİN ETKİSİ ---")
df_test["pay_maj"] = pay_maj
df_test["pay_st2"] = pay_st2
df_test["pay_st1_cal"] = pay_st1_cal
df_test["spread"] = smf_test - ptf_test
df_test["month_str"] = df_test["timestamp"].dt.strftime("%Y-%m")

monthly = df_test.groupby("month_str")[["pay_maj", "pay_st1_cal", "pay_st2", "ptf", "smf"]].mean()
print(monthly)

df_r3 = df_test[r3_mask].copy()
top_1pct_cutoff = df_r3["spread"].quantile(0.99)
extreme_mask = df_r3["spread"] >= top_1pct_cutoff
n_ext = extreme_mask.sum()
print(f"\nRejim 3 içinde en yüksek %1 yayılım eşiği: {top_1pct_cutoff:.2f} TL/MWh ({n_ext} saat)")
print(f"Rejim 3 toplam saatlik ortalama (Açık): {df_r3['pay_maj'].mean():.2f} TL/MWh")
print(f"Rejim 3 (Ekstrem %1 saatler hariç Açık): {df_r3.loc[~extreme_mask, 'pay_maj'].mean():.2f} TL/MWh")
print(f"Rejim 3 Ekstrem %1 saatlerde Açık ortalaması: {df_r3.loc[extreme_mask, 'pay_maj'].mean():.2f} TL/MWh")
print(f"Aşama 2 (Ekstrem %1 hariç): {df_r3.loc[~extreme_mask, 'pay_st2'].mean():.2f} TL/MWh")
print(f"Aşama 2 Ekstrem %1 saatlerde ortalama: {df_r3.loc[extreme_mask, 'pay_st2'].mean():.2f} TL/MWh")

# 6. CANLI EKİM 2026 MODEL SONUÇLARI
live_file = "data/raw/epias_live_202610_verified.parquet"
if os.path.exists(live_file):
    df_live_raw = pd.read_parquet(live_file)
    print(f"\n--- 6. CANLI EKİM 2026 (288 SAAT) DONMUŞ MODEL PERFORMANSI ---")
    df_live_f = df_live_raw.copy().sort_values("timestamp").reset_index(drop=True)
    df_live_f["load_forecast_mw"] = df_live_f["load_forecast_mw"].ffill()
    df_live_f["wind_forecast_mw"] = df_live_f["wind_forecast_mw"].ffill()
    df_live_clean, _, _ = build_thesis_feature_matrix(df_live_f, lag_offset=14)
    
    ptf_l = df_live_clean["ptf"].values
    neg_l = df_live_clean["negativeImbalance"].values
    pos_l = df_live_clean["positiveImbalance"].values
    y_l = df_live_clean["target_direction"].values
    
    live_probs_raw = clf_final.predict_proba(df_live_clean[cols_gop])
    live_probs_cal = calibrator.predict_proba(live_probs_raw)
    act_live_st1 = get_stage1_actions(live_probs_cal, tau=70.0)
    
    live_sp_pred = sp_reg.predict(df_live_clean[cols_dam])
    act_live_st2 = np.where(live_sp_pred > 0, 1, -1)
    
    act_live_maj = np.ones(len(y_l))
    act_live_sur = -np.ones(len(y_l))
    act_live_orc = np.where(y_l == 1, 1, np.where(y_l == -1, -1, 0))
    
    pay_live_orc = calc_exact_pnl(act_live_orc, ptf_l, neg_l, pos_l, y_l)
    pay_live_maj = calc_exact_pnl(act_live_maj, ptf_l, neg_l, pos_l, y_l)
    pay_live_sur = calc_exact_pnl(act_live_sur, ptf_l, neg_l, pos_l, y_l)
    pay_live_st1 = calc_exact_pnl(act_live_st1, ptf_l, neg_l, pos_l, y_l)
    pay_live_st2 = calc_exact_pnl(act_live_st2, ptf_l, neg_l, pos_l, y_l)
    
    print(f"  Toplam Canlı Saat: {len(y_l)}")
    print(f"  Dağılım: Açık={(y_l==1).sum()} (%{(y_l==1).mean()*100:.1f}), Fazla={(y_l==-1).sum()} (%{(y_l==-1).mean()*100:.1f}), Denge={(y_l==0).sum()} (%{(y_l==0).mean()*100:.1f})")
    print(f"  Oracle Üst Sınır:           {np.mean(pay_live_orc):6.2f} TL/MWh")
    print(f"  Her Zaman Açık (Çoğunluk): {np.mean(pay_live_maj):6.2f} TL/MWh (Ağır Kayıp)")
    print(f"  Her Zaman Fazla (Ters):     {np.mean(pay_live_sur):6.2f} TL/MWh")
    print(f"  Aşama 1 (Donmuş Kalibre):   {np.mean(pay_live_st1):6.2f} TL/MWh")
    print(f"  Aşama 2 (Donmuş Yayılım):   {np.mean(pay_live_st2):6.2f} TL/MWh")
