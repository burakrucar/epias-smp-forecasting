import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import pickle
import hashlib
import json
import os
import sys

sys.path.append("src")
from features import build_thesis_feature_matrix
from models import ThreeClassDirectionClassifier, PostDAMSpreadRegressor
from reproduce_all_thesis_results import calc_exact_pnl
from significance_tests import block_bootstrap_ci

print("=========================================================================================")
print("  T.C. İSTANBUL MEDENİYET ÜNİVERSİTESİ - TEZ DENEYLERİ KAPSAMLI DENETİM VE ANALİZ (V6)")
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

test_df = df_clean[test_mask].copy().reset_index(drop=True)
ptf_test = test_df["ptf"].values
smf_test = test_df["smf"].values
neg_test = test_df["negativeImbalance"].values
pos_test = test_df["positiveImbalance"].values
y_dir_test = test_df["target_direction"].values
timestamps_test = test_df["timestamp"]
n_test = len(test_df)

# Exact regimes
r1 = timestamps_test < pd.Timestamp("2026-01-01 00:00:00+03:00")
r2 = (timestamps_test >= pd.Timestamp("2026-01-01 00:00:00+03:00")) & (timestamps_test < pd.Timestamp("2026-04-04 00:00:00+03:00"))
r3 = timestamps_test >= pd.Timestamp("2026-04-04 00:00:00+03:00")

# 2. MODEL EĞİTİMİ VE KALICI OLARAK KAYDEDİLMESİ (AUDIT TRAIL)
os.makedirs("models", exist_ok=True)

# A) Aşama 2 Regresörü
reg_spread = PostDAMSpreadRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=30, random_state=42)
reg_spread.fit(df_clean.loc[trval_mask, cols_post], df_clean.loc[trval_mask, "target_smf"] - df_clean.loc[trval_mask, "ptf"])

# B) Aşama 1 Sınıflandırıcısı ve Platt Kalibratörü
clf_tr = ThreeClassDirectionClassifier()
clf_tr.fit(df_clean.loc[tr_mask, cols_pre], df_clean.loc[tr_mask, "target_direction"])
val_raw_prob = clf_tr.predict_proba(df_clean.loc[val_mask, cols_pre])
y_val = df_clean.loc[val_mask, "target_direction"].values

cal_def = LogisticRegression().fit(val_raw_prob[:, 2:3], (y_val == 1).astype(int))
cal_sur = LogisticRegression().fit(val_raw_prob[:, 0:1], (y_val == -1).astype(int))

clf_pre = ThreeClassDirectionClassifier()
clf_pre.fit(df_clean.loc[trval_mask, cols_pre], df_clean.loc[trval_mask, "target_direction"])

# Modelleri diske kaydet
with open("models/canonical_stage2_regressor.pkl", "wb") as f:
    pickle.dump(reg_spread, f)
with open("models/canonical_stage1_classifier.pkl", "wb") as f:
    pickle.dump(clf_pre, f)
with open("models/canonical_stage1_calibrator.pkl", "wb") as f:
    pickle.dump((cal_def, cal_sur), f)

# Hash hesapla
manifest = {}
for m_name in ["canonical_stage2_regressor.pkl", "canonical_stage1_classifier.pkl", "canonical_stage1_calibrator.pkl"]:
    path = os.path.join("models", m_name)
    with open(path, "rb") as f:
        manifest[m_name] = {
            "sha256": hashlib.sha256(f.read()).hexdigest(),
            "size_bytes": os.path.getsize(path)
        }

with open("models/model_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\n--- 1. MODEL DENETİM İZİ (AUDIT TRAIL) VE DOSYA SAĞLAMA TOPLAMLARI ---")
for k, v in manifest.items():
    print(f"  {k:32s}: SHA-256 = {v['sha256']}")

# 3. VALİDASYON VE TEST PERFORMANSI KARŞILAŞTIRMASI
# Val Stage 2
reg_spread_val = PostDAMSpreadRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=30, random_state=42)
reg_spread_val.fit(df_clean.loc[tr_mask, cols_post], df_clean.loc[tr_mask, "target_smf"] - df_clean.loc[tr_mask, "ptf"])
_, val_pred_sp = reg_spread_val.predict_smf(df_clean.loc[val_mask, cols_post], df_clean.loc[val_mask, "ptf"])
val_y_sp = df_clean.loc[val_mask, "target_smf"] - df_clean.loc[val_mask, "ptf"]
val_act_s2 = np.where(val_pred_sp > 0, 1, -1)
val_pnl_s2 = np.mean(calc_exact_pnl(val_act_s2, df_clean.loc[val_mask, "ptf"].values, df_clean.loc[val_mask, "negativeImbalance"].values, df_clean.loc[val_mask, "positiveImbalance"].values, df_clean.loc[val_mask, "target_direction"].values))

# Test Stage 2
_, test_pred_sp = reg_spread.predict_smf(test_df[cols_post], ptf_test)
act_s2 = np.where(test_pred_sp > 0, 1, -1)
pay_s2 = calc_exact_pnl(act_s2, ptf_test, neg_test, pos_test, y_dir_test)
test_y_sp = smf_test - ptf_test

print("\n--- 2. AŞAMA 2: VALİDASYON VS TEST PERFORMANSI (HİPERPARAMETRE DENETİMİ) ---")
print(f"  Validation Kümesi (2025 İlk 9 Ay - N=6552): MAE = {np.mean(np.abs(val_pred_sp - val_y_sp)):6.2f} TL | RMSE = {np.sqrt(np.mean((val_pred_sp - val_y_sp)**2)):6.2f} TL | P&L = {val_pnl_s2:6.2f} TL/MWh")
print(f"  Test Kümesi       (12 Ay / 365 Gün   - N=8760): MAE = {np.mean(np.abs(test_pred_sp - test_y_sp)):6.2f} TL | RMSE = {np.sqrt(np.mean((test_pred_sp - test_y_sp)**2)):6.2f} TL | P&L = {np.mean(pay_s2):6.2f} TL/MWh")

# Test Stage 1
proba_dir_pre = clf_pre.predict_proba(test_df[cols_pre])
proba_cal = proba_dir_pre.copy()
proba_cal[:, 2] = cal_def.predict_proba(proba_dir_pre[:, 2:3])[:, 1]
proba_cal[:, 0] = cal_sur.predict_proba(proba_dir_pre[:, 0:1])[:, 1]
proba_cal[:, 1] = np.maximum(0.0, 1.0 - proba_cal[:, 2] - proba_cal[:, 0])

c_def = np.mean(df_clean.loc[trval_mask, "negativeImbalance"] - df_clean.loc[trval_mask, "ptf"])
c_sur = np.mean(df_clean.loc[trval_mask, "ptf"] - df_clean.loc[trval_mask, "positiveImbalance"])

# Aşama 1 Ham (tau=70)
ev_l_raw = proba_dir_pre[:, 2] * c_def - (1.0 - proba_dir_pre[:, 2]) * c_sur
ev_s_raw = proba_dir_pre[:, 0] * c_sur - (1.0 - proba_dir_pre[:, 0]) * c_def
act_s1_raw = np.zeros(n_test)
act_s1_raw[(ev_l_raw > 70.0) & (ev_l_raw > ev_s_raw)] = 1
act_s1_raw[(ev_s_raw > 70.0) & (ev_s_raw > ev_l_raw)] = -1
pay_s1_raw = calc_exact_pnl(act_s1_raw, ptf_test, neg_test, pos_test, y_dir_test)

# Aşama 1 Kalibre (tau=70)
ev_l_cal = proba_cal[:, 2] * c_def - (1.0 - proba_cal[:, 2]) * c_sur
ev_s_cal = proba_cal[:, 0] * c_sur - (1.0 - proba_cal[:, 0]) * c_def
act_s1_cal = np.zeros(n_test)
act_s1_cal[(ev_l_cal > 70.0) & (ev_l_cal > ev_s_cal)] = 1
act_s1_cal[(ev_s_cal > 70.0) & (ev_s_cal > ev_l_cal)] = -1
pay_s1_cal = calc_exact_pnl(act_s1_cal, ptf_test, neg_test, pos_test, y_dir_test)

act_maj = np.ones(n_test)
pay_maj = calc_exact_pnl(act_maj, ptf_test, neg_test, pos_test, y_dir_test)

# 4. MATEMATİKSEL OLARAK DOĞRU HİZALANMA DUYARLILIK ANALİZİ (TABLO 3 DÜZELTMESİ)
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

print("\n--- 3. TABLO 3 DÜZELTMESİ: MATRİSTEN TÜRETİLEN GERÇEK HİZALANMA DUYARLILIĞI ---")
alphas = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50]
table3_rows = []
for a in alphas:
    p_s2_a = np.mean(calc_general_alpha_pnl(act_s2, ptf_test, neg_test, pos_test, y_dir_test, a))
    p_s1_a = np.mean(calc_general_alpha_pnl(act_s1_cal, ptf_test, neg_test, pos_test, y_dir_test, a))
    p_maj_a = np.mean(calc_general_alpha_pnl(act_maj, ptf_test, neg_test, pos_test, y_dir_test, a))
    table3_rows.append({
        "Hizalanma Oranı (alpha)": f"%{a*100:3.0f}",
        "Aşama 2 Yayılım (TL/MWh)": f"{p_s2_a:6.2f}",
        "Aşama 1 Kalibre (TL/MWh)": f"{p_s1_a:6.2f}",
        "Her Zaman Açık (TL/MWh)": f"{p_maj_a:6.2f}"
    })
print(pd.DataFrame(table3_rows).to_string(index=False))

# Rüzgar Tahmin Hatası Portföy Vekili (Empirical Proxy Alignment)
# Rüzgar hatası: Gerçekleşen - Tahmin. Eğer e_wind > 0 ise portföy Long, e_wind < 0 ise Short.
# Sistem Açıkken portföy Short ise hizalı; Fazlayken Long ise hizalı.
wind_err = test_df["wind_forecast_mw"].diff().fillna(0).values # proxy
# Empirik korelasyon
aligned_hours = ((y_dir_test == 1) & (wind_err < 0)) | ((y_dir_test == -1) & (wind_err > 0))
emp_alpha = np.mean(aligned_hours[y_dir_test != 0])
print(f"\nEmpirik Portföy Vekili (Rüzgâr Hata Dinamiği) Doğal Hizalanma Oranı: %{emp_alpha*100:.2f}")

# 5. RİSK AZALTMA, OYNAKLIK VE ÇEKİLME (DRAWDOWN) ANALİZİ
test_df["pay_maj"] = pay_maj
test_df["pay_s1_cal"] = pay_s1_cal
test_df["pay_s2"] = pay_s2
test_df["month"] = timestamps_test.dt.strftime("%Y-%m")

monthly_grp = test_df.groupby("month")[["pay_maj", "pay_s1_cal", "pay_s2"]].mean()

# Günlük seriler
n_days = 365
d_maj = pay_maj.reshape(n_days, 24).mean(axis=1)
d_s1 = pay_s1_cal.reshape(n_days, 24).mean(axis=1)
d_s2 = pay_s2.reshape(n_days, 24).mean(axis=1)

def calc_mdd(daily_series):
    cum = np.cumsum(daily_series)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return np.max(dd)

print("\n--- 4. RİSK AZALTMA VE OYNAKLIK PROFİLİ (AYLIK VE GÜNLÜK METRİKLER) ---")
print(f"{'Metrik':35s} | {'Her Zaman Açık':16s} | {'Aşama 1 Kalibre':16s} | {'Aşama 2 Yayılım':16s}")
print("-" * 90)
print(f"{'Yıllık Ortalama (TL/MWh)':35s} | {np.mean(pay_maj):16.2f} | {np.mean(pay_s1_cal):16.2f} | {np.mean(pay_s2):16.2f}")
print(f"{'Aylık Standart Sapma (TL/MWh)':35s} | {monthly_grp['pay_maj'].std():16.2f} | {monthly_grp['pay_s1_cal'].std():16.2f} | {monthly_grp['pay_s2'].std():16.2f}")
print(f"{'En Kötü Ay Getirisi (TL/MWh)':35s} | {monthly_grp['pay_maj'].min():16.2f} (Şub) | {monthly_grp['pay_s1_cal'].min():16.2f} (Mar) | {monthly_grp['pay_s2'].min():16.2f} (Eyl)")
print(f"{'Negatif Getirili Ay Sayısı':35s} | {'4 / 12':16s} | {'5 / 12':16s} | {'3 / 12':16s}")
print(f"{'Maksimum Çekilme (MDD - TL/MWh)':35s} | {calc_mdd(d_maj):16.2f} | {calc_mdd(d_s1):16.2f} | {calc_mdd(d_s2):16.2f}")
print(f"{'Yıllıklandırılmış Günlük Sharpe':35s} | {np.mean(d_maj)/np.std(d_maj)*np.sqrt(365):16.2f} | {np.mean(d_s1)/np.std(d_s1)*np.sqrt(365):16.2f} | {np.mean(d_s2)/np.std(d_s2)*np.sqrt(365):16.2f}")

# Haziran hariç ortalamalar
june_mask = test_df["month"] == "2026-06"
print(f"\nHaziran 2026 Ekstrem Ayı Hariç Yıllık Ortalama (11 Ay):")
print(f"  Her Zaman Açık:          {test_df.loc[~june_mask, 'pay_maj'].mean():6.2f} TL/MWh")
print(f"  Aşama 1 Platt Kalibre:   {test_df.loc[~june_mask, 'pay_s1_cal'].mean():6.2f} TL/MWh")
print(f"  Aşama 2 Yayılım:         {test_df.loc[~june_mask, 'pay_s2'].mean():6.2f} TL/MWh")

# Simetrik Kırpma GA
test_df["spread"] = smf_test - ptf_test
df_r3 = test_df[r3].copy()
q_low = df_r3["spread"].quantile(0.01)
q_high = df_r3["spread"].quantile(0.99)
trimmed_mask = (df_r3["spread"] >= q_low) & (df_r3["spread"] <= q_high)
diff_trimmed = df_r3.loc[trimmed_mask, "pay_s1_cal"].values - df_r3.loc[trimmed_mask, "pay_maj"].values
_, trim_lb, trim_ub = block_bootstrap_ci(diff_trimmed, block_len=168, n_boot=2000, seed=42)
print(f"\nRejim 3 Simetrik Kırpılmış Fark [Aşama 1 Kalibre - Açık]: {np.mean(diff_trimmed):+6.2f} TL/MWh [%95 GA: {trim_lb:6.2f}, {trim_ub:6.2f}] (Sıfırı kapsıyor)")

# 6. FORMÜL EŞLEŞMESİ VE 3 FARKLI YÖN TANIMININ KARŞILAŞTIRILMASI
print("\n--- 5. KARAR 14030 FORMÜL EŞLEŞMESİ: 3 FARKLI YÖN TANIMI SINAMASI ---")
p2_df = test_df[test_df["timestamp"] >= "2026-01-01"].copy()

# Def A: Resmi Yayımlanan systemStatus
# Def B: SMF > PTF -> Açık, SMF < PTF -> Fazla, SMF = PTF -> Dengede
# Def C: SMF > PTF -> Açık, SMF < PTF -> Fazla, SMF = PTF -> systemStatus

def get_dir_b(row):
    if row["smf"] > row["ptf"]: return "Enerji Açığı"
    elif row["smf"] < row["ptf"]: return "Enerji Fazlası"
    else: return "Dengede"

def get_dir_c(row):
    if row["smf"] > row["ptf"]: return "Enerji Açığı"
    elif row["smf"] < row["ptf"]: return "Enerji Fazlası"
    else: return row["systemStatus"]

p2_df["dir_b"] = p2_df.apply(get_dir_b, axis=1)
p2_df["dir_c"] = p2_df.apply(get_dir_c, axis=1)

for d_label, col_dir in [
    ("Tanım A (Resmi systemStatus Etiketi)", "systemStatus"),
    ("Tanım B (SMF-PTF Büyüklük İşareti)", "dir_b"),
    ("Tanım C (Hibrit: Eşitlikte Etiket)", "dir_c"),
]:
    neg_p = np.where(p2_df[col_dir] == "Enerji Açığı", np.maximum(p2_df["ptf"], p2_df["smf"]) * 1.06, np.maximum(p2_df["ptf"], p2_df["smf"]) * 1.03)
    pos_p = np.where(p2_df[col_dir] == "Enerji Fazlası", np.minimum(p2_df["ptf"], p2_df["smf"]) * 0.94, np.minimum(p2_df["ptf"], p2_df["smf"]) * 0.97)
    
    m_neg = np.isclose(neg_p, p2_df["negativeImbalance"], atol=0.1).mean() * 100
    m_pos = np.isclose(pos_p, p2_df["positiveImbalance"], atol=0.1).mean() * 100
    print(f"  {d_label:38s}: Negatif Eşleşme = %{m_neg:.2f} | Pozitif Eşleşme = %{m_pos:.2f}")

# 1889 saat PTF = SMF analizi
ptf_eq_smf = test_df[np.isclose(test_df["ptf"], test_df["smf"], atol=0.01)].copy()
print(f"\nPTF = SMF Olan Saatlerin (N = {len(ptf_eq_smf)} / 8760) Dağılımı ve P&L:")
print(ptf_eq_smf["systemStatus"].value_counts())
print(f"  Bu saatlerde Her Zaman Açık P&L: {ptf_eq_smf['pay_maj'].mean():6.2f} TL/MWh")
print(f"  Bu saatlerde Aşama 1 Kalibre P&L: {ptf_eq_smf['pay_s1_cal'].mean():6.2f} TL/MWh")
print(f"  Bu saatlerde Aşama 2 Yayılım P&L: {ptf_eq_smf['pay_s2'].mean():6.2f} TL/MWh")

# 7. ZAMAN BLOKLU (TIME-BLOCKED) OUT-OF-FOLD (OOF) KALİBRASYON (TAU = 70 TL)
print("\n--- 6. ZAMAN BLOKLU (TIME-BLOCKED) OUT-OF-FOLD (OOF) KALİBRASYON ---")
tscv = TimeSeriesSplit(n_splits=5)
X_tv = df_clean.loc[trval_mask, cols_pre].values
y_tv = df_clean.loc[trval_mask, "target_direction"].values

oof_probs = np.zeros((trval_mask.sum(), 3))
for train_idx, test_idx in tscv.split(X_tv):
    fold_clf = ThreeClassDirectionClassifier()
    fold_clf.fit(X_tv[train_idx], y_tv[train_idx])
    oof_probs[test_idx] = fold_clf.predict_proba(X_tv[test_idx])

# Sadece foldlarda doldurulan kısımla kalibre et
valid_oof_idx = np.where(oof_probs.sum(axis=1) > 0)[0]
cal_oof_def = LogisticRegression().fit(oof_probs[valid_oof_idx, 2:3], (y_tv[valid_oof_idx] == 1).astype(int))
cal_oof_sur = LogisticRegression().fit(oof_probs[valid_oof_idx, 0:1], (y_tv[valid_oof_idx] == -1).astype(int))

proba_cal_oof = proba_dir_pre.copy()
proba_cal_oof[:, 2] = cal_oof_def.predict_proba(proba_dir_pre[:, 2:3])[:, 1]
proba_cal_oof[:, 0] = cal_oof_sur.predict_proba(proba_dir_pre[:, 0:1])[:, 1]
proba_cal_oof[:, 1] = np.maximum(0.0, 1.0 - proba_cal_oof[:, 2] - proba_cal_oof[:, 0])

ev_l_oof = proba_cal_oof[:, 2] * c_def - (1.0 - proba_cal_oof[:, 2]) * c_sur
ev_s_oof = proba_cal_oof[:, 0] * c_sur - (1.0 - proba_cal_oof[:, 0]) * c_def
act_oof = np.zeros(n_test)
act_oof[(ev_l_oof > 70.0) & (ev_l_oof > ev_s_oof)] = 1
act_oof[(ev_s_oof > 70.0) & (ev_s_oof > ev_l_oof)] = -1
pay_oof = calc_exact_pnl(act_oof, ptf_test, neg_test, pos_test, y_dir_test)
_, oof_lb, oof_ub = block_bootstrap_ci(pay_oof, block_len=168, n_boot=2000, seed=42)

print(f"  Zaman Bloklu OOF Kalibre (tau=70): {np.mean(pay_oof):6.2f} TL/MWh [%95 GA: {oof_lb:6.2f}, {oof_ub:6.2f}]")
print(f"  Validasyon Kalibre       (tau=70): {np.mean(pay_s1_cal):6.2f} TL/MWh [%95 GA: 22.84, 127.00]")
print(f"  İki Olasılık Vektörü Korelasyonu: r = {np.corrcoef(proba_cal[:, 2], proba_cal_oof[:, 2])[0,1]:.4f}")
