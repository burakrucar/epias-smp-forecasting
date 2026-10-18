# Türkiye Elektrik Piyasası SMF ve Dengeleme Yönü Tahmini Tez Projesi

## 📖 Proje Özeti
Bu proje, **Türkiye Elektrik Piyasası (EPİAŞ / TEİAŞ)** dinamiklerinde **Sistem Marjinal Fiyatı (SMF - SMP)** ve **Dengeleme Güç Piyasası (DGP) Sistem Yönü**'nün gün öncesi ufukta (**12 ila 36 saat öncesinden saatlik bazda**) bütünleşik (hibrit) olarak tahmin edilmesi ve finansal etkilerinin değerlendirilmesi amacıyla geliştirilmektedir.

---

## 🎯 Temel Hedefler
1. **Sistem Denge Yönü Tahmini (Sınıflandırma):**
   - **Sistem Açığı (+1):** Talep > Arz $\to$ TEİAŞ Yük Alma (YAL) talimatı verir $\to SMF \ge PTF$.
   - **Sistem Fazlası (-1):** Talep < Arz $\to$ TEİAŞ Yük Atma (YAT) talimatı verir $\to SMF \le PTF$.
   - **Dengede (0):** $SMF = PTF$.
2. **Sistem Marjinal Fiyatı (SMF) Tahmini (Regresyon):**
   - TL/MWh cinsinden saatlik fiyat tahmini.
3. **Piyasa Kısıtlı Hibrit / Two-Stage Model:**
   - Yön olasılıklarını fiyat modeline girdi yaparak piyasa kurallarına ($SMF \ge PTF$ veya $SMF \le PTF$) uygun tutarlı tahminler üretmek.
4. **Finansal Simülasyon (P&L / Portföy Arbitrajı):**
   - Model tahminlerinin sanal bir piyasa katılımcısı (üretici/tüketici) portföyünde dengesizlik maliyetlerini ne kadar azalttığının simülasyonu.

---

## ⏱️ Tahmin Ufku (Forecasting Horizon)
- **Kural:** Her gün ($D$) saat 12:00/12:30 itibarıyla ertesi günün ($D+1$) 00:00 - 24:00 saatlik dönemi (yani $t+12$ ile $t+36$ saat arası).
- **Zaman Serisi Bölümleme:** Walk-forward (genişleyen pencere) ile veri sızıntısız (no data leakage) geriye dönük test.

---

## 📂 Proje Dizin Yapısı
```
smp_forecasting_tez/
├── data/
│   ├── raw/             # EPİAŞ'tan çekilen ham veriler
│   └── processed/       # Temizlenmiş ve özellik mühendisliği yapılmış veri setleri
├── src/
│   ├── data_loader.py   # EPİAŞ / eptr2 veri çekme ve işleme fonksiyonları
│   ├── features.py      # Gecikmeler (lags), RES/GES oranları, takvim özellikleri
│   ├── models.py        # LightGBM, XGBoost, CatBoost ve Two-Stage Hibrit model
│   ├── metrics.py       # WAPE, MAE, RMSE, Macro-F1 ve Dengesizlik Backtest
│   └── evaluate.py      # Karşılaştırmalı benchmark ve raporlama scripti
├── notebooks/           # Keşifsel Veri Analizi (EDA) ve deney defterleri
├── reports/
│   ├── figures/         # Tez için grafikler (korelasyon, SHAP, tahmin vs gerçek)
│   └── tables/          # Tez için LaTeX/Markdown performans tabloları
├── .env.example         # EPİAŞ Şeffaflık Platformu API anahtarları şablonu
└── README.md
```

---

## 🚀 Hızlı Başlangıç
1. EPİAŞ Şeffaflık Platformu kullanıcı bilgilerini `.env` içine girin:
   ```env
   EPTR_USERNAME=epias_kullanici_adiniz
   EPTR_PASSWORD=epias_sifreniz
   ```
2. Gerekli kütüphaneler: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `eptr2`, `matplotlib`, `seaborn`.
