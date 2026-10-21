# EPİAŞ Elektrik Piyasasında SMF ve Sistem Denge Yönü Tahmini
## Ekonometrik Doğrulama, Model Atıf Analizi ve İleriye Dönük Test Protokolü

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Protocol: protocol--v1](https://img.shields.io/badge/Protocol-protocol--v1-success.svg)](reports/prospective_evaluation_protocol_2026_2027.md)
[![Author: Burak Ramazan Uçar](https://img.shields.io/badge/Author-Burak%20Ramazan%20U%C3%A7ar-blue.svg)](https://github.com/burakrucar)

> **Geliştirici & Araştırmacı:** Burak Ramazan Uçar  
> **Proje:** Türkiye Elektrik Piyasası Algoritmik Fiyat & Denge Yönü Tahminleme ve Arbitraj Çerçevesi  

---

## 📌 Proje Hakkında

Bu proje; **Türkiye Elektrik Piyasası'nda (EPİAŞ GÖP & TEİAŞ DGP)** Sistem Marjinal Fiyatı (**SMF**) ile Sistem Denge Yönü'nün (**$D_t$**) makine öğrenmesi modelleriyle tahmin edilmesini, elde edilen tahminlerin finansal getiriye (**P&L / Arbitraj**) dönüştürülmesini ve makine öğrenmesi literatüründeki *p-hacking / lookahead bias* tuzaklarını ortadan kaldıran yüksek standartlı bir **Kayıtlı Rapor (Registered Report / Pre-Analysis Plan)** protokolünü içermektedir.

### 🎯 Temel Araştırma Sorusu
> *"Gelişmiş makine öğrenmesi modelleri (LightGBM, Platt kalibrasyonu), elektrik piyasasında sistem yönünü ve SMF'yi tahmin ederek piyasanın en basit kuralı olan 'Her Zaman Açık' stratejisine karşı istatistiki olarak anlamlı bir **ekonomik üstünlük (alfa)** sağlayabilir mi?"*

---

## ⚡ Piyasa Mekanizması ve İki Aşamalı Karar Mimarisi

Elektrik piyasasında zaman kısıtları ve bilgi akışı asimetriktir. Model geriye dönük bilgi sızıntısını (**lookahead bias**) kesin biçimde önleyecek iki ayrı karar anına göre tasarlanmıştır:

```
[D-1 Günü (Teslimattan 1 Gün Önce)]                         [D Günü (Fiziksel Teslimat)]
 Saat 11:30                  Saat 12:30    Saat 14:00        Saat 00:00 ---------- 23:00
      |                           |             |                 |                      |
      v                           v             v                 v                      v
[AŞAMA 1 KARAR ANI]        [GÖP Kapanışı] [AŞAMA 2 KARAR ANI] [24 Saatlik Teslimat Dönemi]
 • PTF HENÜZ BİLİNMİYOR      • Teklifler   • PTF ARTIK BİLİNİYOR  • SMF ve Denge Yönü
 • Tahmin Ufku: 12-36 saat     Kilitlenir  • Tahmin Ufku: 10-34h    fiziksel olarak oluşur
 • Hedef: 3 Sınıflı Yön                    • Hedef: Yayılım (SMF-PTF)
```

1. **Aşama 1 (GÖP Öncesi — Saat 11:30):** $PTF$ henüz bilinmemektedir. Yalnızca tarihsel seriler, rüzgar ve yük tahminleri kullanılarak 3 sınıflı sistem yönü ($D_t \in \{+1, -1, 0\}$) tahmin edilir ve **Platt scaling** ile kalibre edilen olasılıklar üzerinden marjinal beklenen değer kuralı ($\tau = 70$ TL) ile karar verilir.
2. **Aşama 2 (GÖP Sonrası — Saat 14:00):** Ertesi günün kesinleşen saatlik $PTF$ değerleri modele girdi yapılarak $SMF - PTF$ yayılımı (**spread**) regresyonla tahmin edilir.

---

## ⚖️ Finansal Uzlaştırma (EPDK Karar 14030)

Piyasa katılımcılarının nakit akışını belirleyen cezalı uzlaştırma formülleri ($k=l=0,06$):
* **Negatif Dengesizlik Fiyatı ($P_{\text{neg}}$):** $P_{\text{neg}} = \max(PTF, SMF) \times (1 + k)$
* **Pozitif Dengesizlik Fiyatı ($P_{\text{pos}}$):** $P_{\text{pos}} = \min(PTF, SMF) \times (1 - l)$

| Pozisyon ($a_t$) | Sistem Açıkta ($D_t = +1$) | Sistem Fazlada ($D_t = -1$) |
| :--- | :---: | :---: |
| **Uzun ($a_t = +1$)** | $+(P_{\text{neg}} - PTF)$ *(Kâr / Tasarruf)* | $-(PTF - P_{\text{pos}})$ *(Zarar / İndirim kaybı)* |
| **Kısa ($a_t = -1$)** | $-(P_{\text{neg}} - PTF)$ *(Ceza / Zarar)* | $+(PTF - P_{\text{pos}})$ *(Kâr / Satış kazancı)* |
| **Pasif ($a_t = 0$)** | $0,00$ TL | $0,00$ TL |

---

## 🔬 Araştırmanın Ortaya Koyduğu 7 Temel Bulgusu

12 aylık (8.760 saat) test kümesi üzerinde yapılan ekonometrik analizlerin öne çıkan sonuçları:

1. **PTF, SMF'nin Baskın Bilgi Çapasıdır ($L_1$ vs $L_2$):** Saatlerin **%21,14'ünde** $SMF=PTF$ gerçekleşir. Naif $SMF=PTF$ tahmini mutlak hatada (MAE: **682,75 TL**) makine öğrenmesini (**748,14 TL**) geçer. Modelin RMSE'deki üstünlüğü istatistiki olarak anlamsızdır ($p = 0,2702$).
2. **Yön Sınıflandırma Becerisi Sınırlıdır:** Aşama 1 modeli yönü **%59,66** doğrulukla tahmin ederek çoğunluk sınıfına (%55,74) göre **+3,92 puanlık** mütevazı bir üstünlük sağlar ($p = 0,034$), ancak akşam zirvesinde bu avantaj sıfırlanır.
3. **P&L Yön Doğruluğuna Değil, Yayılım Büyüklüğüne Bağlıdır:** Yön doğruluğu yüksek olan model (%62,21) P&L'de **zarar ederken (-20,81 TL)**; yön doğruluğu sadece %41,70 olan regresör **kâr eder (+119,14 TL)**! Elektrik piyasasında kazanç yönün sıklığına değil, sıçramanın genliğine bağlıdır.
4. **Ekonomik Üstünlük (Alfa) Sağlanamamıştır:** Aşama 1 modeli (**+67,33 TL**), hiçbir şey bilmeyen "Her Zaman Açık" naif kuralını (**+71,73 TL**) geçememiştir; 7 günlük blok bootstrap farkı istatistiki olarak anlamsızdır ($p > 0,75$).
5. **Tek Aya Bağımlılık (Haziran Etkisi):** Yıllık kârın **%60'tan fazlası** tek bir aydaki (Haziran 2026) aşırı sıçramalardan kaynaklanır. Haziran dışlandığında stratejiler eşitlenir ($p = 0,184$).
6. **Risk Azalması Algoritmik Değil, Mekaniktir:** Oynaklıktaki %28'lik düşüş "akıllı filtreleme" değil, saatlerin %51'inde pasif kalarak piyasa maruziyetinin yarıya indirilmesinin mekanik sonucudur.
7. **Seviye Etkisi Aldatmacası:** Aşama 2'nin kazancı karmaşık ML sezgisinden değil, fiyat tavan/tabanının dayattığı ortalamaya dönüş kuralından gelir. Yalnızca eğitim medyanına dayanan tek satırlık kural (*"PTF < 2.500 TL ise Uzun kal"*), tek başına **+140,57 TL/MWh** kâr üretir!

---

## 📊 P&L Atıf Dekompozisyonu (Tablo 4D)

$$\text{Net P\&L} = \text{Saf Yayılım} + (\text{Ceza Marjı} + \text{Etiket Uyumsuzluğu}) + \text{Veri Sapması}$$

* **Aşama 2 Modeli (+119,14 TL):** Kazancının tamamı (+150,50 TL) doğrudan **fiyat yayılımından (spread)** gelir.
* **Her Zaman Açık Kuralı (+71,73 TL):** Kazancının **%80'den fazlası (+68,47 TL)** piyasayı tahmin etmekten değil; mevzuatın sisteme tanıdığı **$k=0,06$ asimetrik ceza marjından** gelir!

---

## 🔒 İleriye Dönük Doğrulama Protokolü (`protocol-v1`)

Veri sızıntısını ve geriye dönük seçilim yanlılığını engellemek adına modeller ve kodlar 20 Ekim 2026'da dondurulmuştur:
* **Protokol Etiketi:** `protocol-v1` (Git Commit: `49ccb61dfeb16815a1fdab580af3f6fd7543de78`)
* **SHA-256 Manifestosu:** [`MANIFEST_SHA256.txt`](MANIFEST_SHA256.txt)
* **İleriye Dönük Test Penceresi:** 21 Ekim 2026 00:00 – 31 Mart 2027 23:00 (**162 Gün / 3.888 Saat**)
* **Birincil İstatistiki Hipotez:** Haftalık bloklu işaret-çevirme permütasyon testi ($\alpha = 0,05$).

---

## 📁 Dizin Yapısı

```
epias-smp-forecasting/
├── data/
│   ├── raw/                  # EPİAŞ resmi piyasa verileri (Parquet)
│   └── processed/            # Master tahmin tablosu
├── models/
│   ├── canonical_stage1_classifier.pkl  # Dondurulmuş Aşama 1 LightGBM modeli
│   ├── canonical_stage1_calibrator.pkl  # Dondurulmuş Platt kalibratörü
│   ├── canonical_stage2_regressor.pkl   # Dondurulmuş Aşama 2 yayılım regresörü
│   └── model_manifest.json              # Model hiperparametreleri ve metaveri
├── reports/
│   ├── figures/              # Yüksek çözünürlüklü grafikler ve görselleştirmeler
│   ├── tables/               # Ekonometrik test ve karşılaştırma tabloları (CSV)
│   ├── prospective_evaluation_protocol_2026_2027.md # Dondurulmuş test protokolü
│   ├── tez_deney_sonuclari_raporu.md    # Kapsamlı deney ve doğrulama raporu
│   └── tez_konusu_ve_calisma_rehberi.md # Detaylı model ve piyasa dokümantasyonu
├── scripts/
│   └── build_html_report_v5.py          # Rapor derleme araçları
├── src/
│   ├── run_master_thesis_v5.py          # Master doğrulama ve tablo üretim pipeline'ı
│   ├── models.py / features.py          # Model mimarisi ve öznitelik mühendisliği
│   └── metrics.py                       # Finansal P&L ve ekonometrik metrikler
├── MANIFEST_SHA256.txt                  # Kriptografik mühürleme dosyası
├── requirements.txt                     # Bağımlılıklar ve kütüphane sürümleri
└── README.md
```

---

## 🚀 Kurulum ve Çalıştırma

```bash
# Depoyu klonlayın
git clone https://github.com/burakrucar/epias-smp-forecasting.git
cd epias-smp-forecasting

# Sanal ortamı kurun ve kütüphaneleri yükleyin
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Master pipeline'ı çalıştırarak tüm tabloları ve sonuçları üretin
python3 src/run_master_thesis_v5.py

# Dondurulan modellerin kriptografik bütünlüğünü doğrulayın
shasum -a 256 -c MANIFEST_SHA256.txt
```

---

## 👤 Geliştirici & İletişim

* **Burak Ramazan Uçar**
* **GitHub:** [@burakrucar](https://github.com/burakrucar)
* **E-posta:** burakrucar@gmail.com
