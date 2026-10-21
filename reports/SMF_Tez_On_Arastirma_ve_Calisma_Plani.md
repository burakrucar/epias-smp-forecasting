# TÜRKİYE ELEKTRİK PİYASASINDA SİSTEM MARJİNAL FİYATI (SMF) VE DENGELEME YÖNÜ TAHMİNİ
## LİSANS BİTİRME TEZİ: ÖN ARAŞTIRMA, METODOLOJİK ÇERÇEVE VE ÇALIŞMA PLANI

**Öğrenci:** Burak Ramazan Uçar  
**Bölüm:** Matematik Bölümü Lisans Programı  
**Tarih:** Ekim 2026  
**Doküman Türü:** Tez Danışmanı Görüşme ve Ön Araştırma Dosyası  

---

## İÇİNDEKİLER

1. [Giriş ve Araştırmanın Matematiksel Motivasyonu](#1-giriş-ve-araştırmanın-matematiksel-motivasyonu)
2. [Türkiye Elektrik Piyasası Mekanizması ve Temel Kavramlar](#2-türkiye-elektrik-piyasası-mekanizması-ve-temel-kavramlar)
3. [Mevzuat ve Finansal Uzlaştırma (EPDK Dengesizlik Fiyatlandırması)](#3-mevzuat-ve-finansal-uzlaştırma-epdk-dengesizlik-fiyatlandırması)
4. [Altın Metodolojik Kural: Veri Sızıntısını (Data Leakage) Önleme Protokolü](#4-altın-metodolojik-kural-veri-sızıntısını-data-leakage-önleme-protokolü)
5. [İki Temel Matematiksel Problem Tanımı](#5-iki-temel-matematiksel-problem-tanımı)
6. [Ekonometrik ve İstatistiki Doğrulama Kurgusu](#6-ekonometrik-ve-istatistiki-doğrulama-kurgusu)
7. [Model Hiyerarşisi: Basitten Karmaşığa](#7-model-hiyerarşisi-basitten-karmaşığa)
8. ["Niye Öyle Oluyor?" Analizi ve Model Açıklanabilirliği (XAI)](#8-niye-öyle-oluyor-analizi-ve-model-açıklanabilirliği-xai)
9. [Danışman Hocaya Sunulacak 7 Soru ve Hazır Öneri Paketi](#9-danışman-hocaya-sunulacak-7-soru-ve-hazır-öneri-paketi)
10. [Haftalık Tez İlerleme Takvimi (Aşama 0 - Aşama 8)](#10-haftalık-tez-ilerleme-takvimi-aşama-0---aşama-8)

---

## 1. Giriş ve Araştırmanın Matematiksel Motivasyonu

Elektrik enerjisi, endüstriyel ölçekte depolanması oldukça maliyetli olan ve iletim şebekesinde arz ile talebin her saniye fiziksel dengede (50 Hz frekans) tutulmasını zorunlu kılan benzersiz bir emtiadır. Şebekede anlık dengesizlik oluşması durumunda iletim sistemi operatörü (Türkiye'de **TEİAŞ**) anlık talimatlarla sisteme müdahale eder.

Bu tez çalışmasında iki temel tahmin problemi ele alınmaktadır:
1. **Regresyon Problemi:** Gelecek $t$ saatindeki **Sistem Marjinal Fiyatı'nın (SMF)** TL/MWh cinsinden tahmini.
2. **Sınıflandırma Problemi:** Sistemin o saatte **Enerji Açığında mı (+1)** yoksa **Enerji Fazlasında mı (-1)** olacağının (Dengeleme Yönü) tahmini.

Bir matematik bölümü bitirme tezi olarak bu çalışmanın özgün değeri; piyasa dinamiklerini stokastik zaman serisi, düzenlileştirilmiş regresyon ve modern gradyan artırma (gradient boosting) algoritmalarıyla modellerken, **kesin sızıntısız (leakage-free)** bir veri protokolü ve **açıklanabilir yapay zeka (SHAP)** analizleriyle matematiksel ve iktisadi gerekçeleri ortaya koymasıdır.

---

## 2. Türkiye Elektrik Piyasası Mekanizması ve Temel Kavramlar

Türkiye elektrik toptan ticareti EPİAŞ ve TEİAŞ koordinasyonunda birbirini takip eden katmanlı piyasalardan oluşur:

```
[D-1 Günü: 12:30]                   [D-1 Günü: 14:00]              [D Günü (00:00 - 23:00)]
       │                                   │                                  │
       ▼                                   ▼                                  ▼
Gün Öncesi Piyasası (GÖP)          PTF Kesinleşir                  Dengeleme Güç Piyasası (DGP)
• Teklifler toplanır               • 24 saatlik referans fiyat     • Anlık fiziksel dengeleme (TEİAŞ)
• Sosyal refah maksimizasyonu        kamuya duyurulur              • YAL / YAT talimatları verilir
• Çıktı: PTF (TL/MWh)                                              • Çıktı: SMF (TL/MWh) ve Yön
```

### Temel Piyasa Bileşenleri
* **GÖP (Gün Öncesi Piyasası):** Fiziksel teslimat gününden bir gün önce ($D-1$) saat 12:30'a kadar katılımcıların teklif verdiği ve saatlik referans fiyatın (**Piyasa Takas Fiyatı - PTF**) belirlendiği organize toptan elektrik piyasasıdır.
* **DGP (Dengeleme Güç Piyasası):** Gerçek zamanlı teslimat anında ($D$ günü) TEİAŞ Ulusal Yük Tevzi Merkezi tarafından şebeke frekansını dengede tutmak için yürütülen piyasadır.
* **YAL (Yük Alma Talimatı):** Tüketimin üretimden fazla olması (enerji açığı) durumunda TEİAŞ'ın santrallere "üretimi artır" talimatı vermesidir.
* **YAT (Yük Atma Talimatı):** Üretimin tüketimden fazla olması (enerji fazlası) durumunda santrallere "üretimi kıs" talimatı verilmesidir.
* **SMF (Sistem Marjinal Fiyatı):** Sistem dengesini sağlamak için devreye giren son (en marjinal) teklifin fiyatıdır.
* **Yayılım (Spread - $S_t$):**
  $$S_t = SMF_t - PTF_t$$
* **Sistem Dengeleme Yönü ($D_t$):**
  $$D_t = \begin{cases} +1 & \text{Enerji Açığı (YAL ağırlıklı)} \implies SMF_t \ge PTF_t \\ -1 & \text{Enerji Fazlası (YAT ağırlıklı)} \implies SMF_t \le PTF_t \\ 0 & \text{Dengede (Nadir durum)} \implies SMF_t = PTF_t \end{cases}$$

---

## 3. Mevzuat ve Finansal Uzlaştırma (EPDK Dengesizlik Fiyatlandırması)

Piyasa katılımcıları (üreticiler, tedarikçiler) gün öncesinde taahhüt ettikleri üretim/tüketim miktarlarından saptıklarında **dengesizliğe (imbalance)** düşerler. EPDK mevzuatına göre finansal uzlaştırma şu formülle yapılır:

* **Enerji Açığı Veren Katılımcı (Eksik Üretim Yapan):**
  $$\text{Ödeme Fiyatı} = \max(PTF_t, SMF_t) \times (1 + k)$$
* **Enerji Fazlası Veren Katılımcı (Fazla Üretim Yapan):**
  $$\text{Satış Geliri Fiyatı} = \min(PTF_t, SMF_t) \times (1 - l)$$

Burada $k$ ve $l$ EPDK tarafından belirlenen ceza katsayılarıdır ($k, l \approx 0.03$).  
Bu formül açıkça gösterir ki; **katılımcının kâr veya zararı, doğrudan $SMF_t$ ile $PTF_t$ arasındaki yayılıma ve sistemin yönüne bağlıdır**. Dolayısıyla yönü ve SMF'yi önceden doğru tahmin eden bir model, doğrudan parasal fayda (arbitraj ve hedging) sağlar.

---

## 4. Altın Metodolojik Kural: Veri Sızıntısını (Data Leakage) Önleme Protokolü

Bir zaman serisi tahmin çalışmasının bilimsel güvenilirliği, **tahmin anında elinde olmayan hiçbir bilginin modele girdi olarak verilmemesine** bağlıdır.

```
[D-1 Günü Saat 12:00 (Tahmin Kesme Anı / Origin)]
═══════════════════════════════════════════════════════════════════════════════════► Zaman
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
   [Geçmiş Veriler (Bilinenler)]             [Gelecek Dönem: D Günü 00:00 - 23:00]
   • Kamuya açıklanmış t-24 öncesi SMF       • Hedef: 24 saatin her biri için SMF ve Yön
   • Kamuya açıklanmış tüketim/üretim        • Tahmin Ufku: 12 saat ile 36 saat ilerisi
   • D günü Yük Tahmini (Forecast)           • D gününün gerçekleşen tüketimi YASAK!
   • D günü Rüzgar/Güneş Tahmini             • D gününün PTF'si YASAK! (12:00'de henüz yok)
```

### Sızıntıyı Önleme Kuralları:
1. **Kesme Saati (Cutoff):** Her gün $D-1$ günü saat 12:00.
2. **Tahmin Ufku ($h$):** $D$ gününün saat 00:00'ı 12 saat ileride, saat 23:00'ı 35-36 saat ileridedir. Ufuk tam olarak $h \in [12, 36]$ saat aralığıdır.
3. **Yasak Girdiler:** 
   - $D$ gününün gerçekleşen tüketim veya hava durumu değerleri (çünkü sadece tahminleri mevcuttur).
   - $D$ gününün PTF değeri (GÖP 12:30'da kapanır, PTF 13:00–14:00 civarı açıklanır; saat 12:00 tahmininde kullanılamaz).
4. **Zaman Bölmesi (Temporal Split):** Veri setinde rastgele karıştırma (random shuffle) veya klasik k-katlı çapraz doğrulama kesinlikle uygulanmaz. Veriler kronolojik sırayla:
   - **Eğitim Kümesi (Train):** İlk dönemler
   - **Doğrulama Kümesi (Validation):** Model seçimi ve hiperparametre ayarı
   - **Test Kümesi (Out-of-sample Test):** En son dönem (örneğin son 6-12 ay) şeklinde ayrılır.

---

## 5. İki Temel Matematiksel Problem Tanımı

### Problem 1: Saatlik SMF Fiyat Tahmini (Regresyon)
* **Hedef:** $\widehat{SMF}_{t+h} = f(X_t)$ veya Yayılım Tahmini $\widehat{S}_{t+h} = g(X_t)$ ($h \in [12, 36]$).
* **Parasal Hata Ölçüleri (Hocanın "TL Olarak Kıyasla" İsteği İçin):**
  - **Ortalama Mutlak Hata (MAE):**
    $$MAE = \frac{1}{N} \sum_{i=1}^N |SMF_i - \widehat{SMF}_i| \quad \text{(TL/MWh cinsinden doğrudan parasal sapma)}$$
  - **Kök Ortalama Kare Hata (RMSE):**
    $$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (SMF_i - \widehat{SMF}_i)^2} \quad \text{(Fiyat sıçramalarına yüksek ceza verir)}$$
  - **Simetrik Ortalama Yüzde Hata (sMAPE):**
    $$sMAPE = \frac{100\%}{N} \sum_{i=1}^N \frac{|SMF_i - \widehat{SMF}_i|}{(|SMF_i| + |\widehat{SMF}_i|)/2}$$
    *(Not: Klasik MAPE, sıfıra yakın fiyatlarda paydayı sıfır yaparak patladığı için literatürde sMAPE tercih edilir).*

### Problem 2: Sistem Denge Yönü Tahmini (Sınıflandırma)
* **Hedef:** $\widehat{D}_{t+h} \in \{+1, -1\}$ (Enerji Açığı vs. Enerji Fazlası).
* **Başarı Ölçüleri:**
  - Doğruluk (Accuracy) & Dengeli Doğruluk (Balanced Accuracy)
  - F1-Skoru ve Hassasiyet (Precision) / Duyarlılık (Recall)
  - Matthews Korelasyon Katsayısı (MCC - Dengesiz sınıflarda en güvenilir istatistiki metrik):
    $$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

---

## 6. Ekonometrik ve İstatistiki Doğrulama Kurgusu

Bir matematik tezi olarak serinin stokastik özellikleri titizlikle test edilecektir:

1. **Durağanlık Analizi (Stationarity):**
   - Augmented Dickey-Fuller (ADF) ve KPSS testleri uygulanarak ham serinin birim kök içerip içermediği belirlenir.
   - Enflasyon ve tavan fiyat artışları nedeniyle seride rejim değişikliği varsa; logaritmik dönüşüm, fark serisi veya PTF'ye göre normalize yayılım ($S_t / PTF_t$) yaklaşımları matematiksel olarak test edilir.
2. **Kıyaslama ve Üstünlük Testi (Diebold-Mariano Testi):**
   - Geliştirilen makine öğrenmesi modelinin baz modellere (Naive) kıyasla elde ettiği hata düşüşünün tesadüf mü yoksa istatistiki olarak anlamlı mı olduğu Diebold-Mariano testi ile kanıtlanır:
   $$DM = \frac{\bar{d}}{\sqrt{\widehat{Var}(\bar{d})}} \sim \mathcal{N}(0, 1)$$

---

## 7. Model Hiyerarşisi: Basitten Karmaşığa

Çalışmada modeller kademeli olarak inşa edilecek ve her birinin katkısı raporlanacaktır:

1. **Baz Modeller (Benchmarks):**
   - *Seasonal Naive (Gecikmeli SMF):* Dünkü aynı saatin değeri ($SMF_{t-24}$) ve geçen haftanın aynı saati ($SMF_{t-168}$).
   - *Yön için Naive:* Dünkü aynı saatin dengeleme yönü.
   - *Tezdeki Rolü:* Karmaşık modeller bu basit kuralları geçemiyorsa bilimsel bir değer üretilememiştir demektir.
2. **İstatistiksel / Doğrusal Modeller:**
   - ARX / SARIMAX (Dışsal değişkenli otoregresif model).
   - Ridge, Lasso, ElasticNet Regresyon (Katsayı büyüklükleri ve değişken seçimi matematiksel olarak yorumlanır).
3. **Ağaç Tabanlı Topluluk Modelleri (Ensemble Learning):**
   - Random Forest Regressor / Classifier
   - LightGBM & XGBoost (Elektrik fiyat tahmin literatüründe en yüksek başarıyı veren doğrusal olmayan algoritmalar).
4. **Sınıflandırma Olasılık Kalibrasyonu:**
   - Lojistik regresyon ve Platt Scaling / Isotonic Regression ile modelin ürettiği sınıf olasılıkları kalibre edilir.

---

## 8. "Niye Öyle Oluyor?" Analizi ve Model Açıklanabilirliği (XAI)

Hocanın "Niye öyle oluyor, sonuçların nedenlerini açıklayın" sorusuna cevap vermek için modeller kara kutu (black box) olarak bırakılmayacaktır:

1. **SHAP (SHapley Additive exPlanations) Analizi:**
   - Oyun teorisine dayalı Shapley değerleri kullanılarak her bir girdinin (rüzgar tahmini, kalan yük, saatlik mevsimsellik) fiyata ve yöne pozitif/negatif katkısı hesaplanacaktır.
2. **Kalan Yük (Residual Load) Etkisi:**
   $$\text{Kalan Yük} = \text{Tahmini Tüketim} - (\text{Tahmini Rüzgar} + \text{Tahmini Güneş})$$
   - Kalan yük arttığında pahalı doğalgaz ve kömür santralleri devreye girmekte, bu da SMF sıçramalarına ve enerji açığına yol açmaktadır. Modelin bu iktisadi kuralı öğrenip öğrenmediği doğrulanacaktır.
3. **Tahmin Ufku Etkisi ($h=12$ vs. $h=35$):**
   - 12 saat sonrası için yapılan tahminin hatası ile 35 saat sonrası için yapılan tahminin hatası karşılaştırılacak; zaman ilerledikçe belirsizliğin nasıl arttığı grafiklerle incelenecektir.

---

## 9. Danışman Hocaya Sunulacak 7 Soru ve Hazır Öneri Paketi

Hoca görüşmesine giderken sadece soru sormak yerine, ekibimizin geliştirdiği **hazır önerilerle** gitmek profesyonel bir duruş sağlar:

| # | Plandaki Soru | Önerdiğimiz Yanıt ve Tez Yaklaşımımız |
|---|---|---|
| **1** | **12–36 saat yorumu doğru mu?** | **Evet.** $D-1$ günü saat 12:00'de tahmin üretilip $D$ gününün 24 saati hedeflendiğinde, 00:00 saati 12 saat, 23:00 saati 35 saat ileride kalmaktadır. Bu tam 12–36 saatlik standart ufuktur. |
| **2** | **SMF ile PTF ilişkisi ne olmalı?** | Hem doğrudan SMF'yi (TL/MWh) hem de PTF sonrası aşamada $SMF - PTF$ farkını (spread) ayrı modellerle tahmin edip kıyaslayacağız. |
| **3** | **Yön sınıflandırması kaç sınıf olmalı?** | Sistemde tam "dengede" durumu (%1'den az) istisnai olduğundan ana sınıflandırma **2 sınıflı (Açık: +1 vs Fazla: -1)** kurulmuş, dengede durumu analizde dipnot olarak ele alınmıştır. |
| **4** | **Hangi tarih aralığı ve test dönemi?** | EPİAŞ Şeffaflık Platformu'ndan 2022–2025/2026 saatlik verileri; son 12 ay dış test (out-of-sample) kümesi olarak ayrılacaktır. |
| **5** | **Tercih edilen başarı ölçüsü nedir?** | Hocanın "parasal kıyaslama" vurgusuna tam uygun olarak **MAE ve RMSE (TL/MWh)** ana metrikler; yüzde hata için sıfırdan etkilenmeyen **sMAPE**; yön için **Balanced Accuracy ve MCC** kullanılacaktır. |
| **6** | **Beklenen model karmaşıklığı nedir?** | Temel istatistiki modeller (ARX, Ridge) + Ağaç tabanlı topluluk modelleri (LightGBM/XGBoost) tezin omurgasını oluşturacaktır. Derin öğrenme (LSTM) isteğe bağlı ek deney olarak tutulacaktır. |
| **7** | **Veri kaynağı ne olacak?** | Yalnızca kamuya açık ve doğrulanabilir veriler: EPİAŞ Şeffaflık Platformu, TEİAŞ Yük Tevzi raporları ve açık hava tahmin servisleri. |

---

## 10. Haftalık Tez İlerleme Takvimi (Aşama 0 - Aşama 8)

* **Hafta 1 (Aşama 0):** Danışman hoca ile plan netleştirme, Git ortamı ve Python paketlerinin kontrolü.
* **Hafta 2–3 (Aşama 1):** Piyasa yapısı ve yerli/yabancı literatürün akademik metne dökülmesi.
* **Hafta 3–4 (Aşama 2):** Veri toplama, zaman etiketleme ve yayın zamanı (publication lag) tablosunun kesinleştirilmesi.
* **Hafta 5–6 (Aşama 3):** Keşifsel Veri Analizi (EDA), durağanlık testleri (ADF/KPSS), aykırı değer ve rejim analizi.
* **Hafta 6–7 (Aşama 4):** Deney tasarımı, temporal split kurgusu ve metrik fonksiyonlarının hazırlanması.
* **Hafta 7–8 (Aşama 5):** Baz modellerin (Seasonal Naive, dünkü yön) koşturulması ve kıyaslama tablosunun tabanının atılması.
* **Hafta 8–11 (Aşama 6):** Makine öğrenmesi modelleri (ARX, Ridge, LightGBM, XGBoost) eğitimi ve hiperparametre optimizasyonu.
* **Hafta 11–12 (Aşama 7):** "Niye öyle oluyor?" hata analizi, SHAP grafikleri, ufuk analizi ve Diebold-Mariano testleri.
* **Hafta 12–14 (Aşama 8):** Tez metninin (Giriş, Yöntem, Bulgular, Tartışma, Sonuç) son haline getirilmesi ve savunma sunumu hazırlığı.

---
*Bu çalışma rehberi, T.C. İstanbul Medeniyet Üniversitesi Matematik Bölümü Lisans Bitirme Tezi standartlarına uygun olarak hazırlanmıştır.*
