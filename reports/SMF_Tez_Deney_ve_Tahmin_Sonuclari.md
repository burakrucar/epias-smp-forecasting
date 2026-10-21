# TÜRKİYE ELEKTRİK PİYASASINDA SMF VE DENGELEME YÖNÜ TAHMİNİ
## AŞAMA 2 - AŞAMA 7 DENEYSEL DOĞRULAMA VE MODEL TAHMİN SONUÇLARI RAPORU (REVİZYON V5)

**Öğrenci:** Burak Ramazan Uçar  
**Bölüm:** Matematik Bölümü Lisans Programı  
**Tarih:** 21 Ekim 2026 (Revizyon V5 - Tek Merkezli Tahmin Parquet'si ve Programatik Doğrulama)  
**Çalışma Kimliği (Run ID):** `run_1792585978`  
**Tahmin Veri Kaynağı:** `data/processed/test_predictions_master.parquet` (8.760 saatlik tek ana dosya)  
**Veri Kaynağı:** EPİAŞ Şeffaflık Platformu Resmi Piyasa Veri Seti  
**Zaman Kapsamı:**  
* **Ham Veri:** 24.096 saat / 1.004 gün (01.01.2024 - 30.09.2026)  
* **Kullanılabilir Veri:** 23.424 saat / 976 gün (29.01.2024 - 30.09.2026; 4 haftalık / 672 saatlik gecikme ısınma penceresi arındırılmıştır)  
* **Eğitim Kümesi (Train):** 29.01.2024 - 30.09.2024 (5.904 saat)  
* **Doğrulama Kümesi (Validation):** 01.10.2024 - 30.09.2025 (8.760 saat / 365 gün)  
* **Eğitim + Doğrulama (Full Train):** 29.01.2024 - 30.09.2025 (14.664 saat; test tahminlerini üreten nihai modeller bu birleşik veriyle eğitilmiştir)  
* **Dış Test Kümesi (Test):** 01.10.2025 - 30.09.2026 (8.760 saat / 365 gün)  
**Test Dönemi Fiyat Seviyesi:** Ortalama SMF = **2.198,25 TL/MWh** | Medyan SMF = **2.598,51 TL/MWh**  

---

## 1. YÖNTEM, ZAMAN ÇİZELGESİ VE SIZINTISIZ ÖZELLİK MİMARİSİ

### A. Piyasa Zaman Çizelgesi ve Kesme Anı Kurgusu
Ana tahmin kesme anı teslimat gününden bir gün önce ($D-1$) saat **12:00** olarak tanımlanmıştır. Hedef dönem teslimat gününün ($D$) 24 saatidir ($h = 12, 13, \dots, 35$ saatlik gerçek tahmin ufku).

* **SMF ve Sistem Yönü Yayın Kuralı:** EPİAŞ piyasa mekanizmasında SMF ve Sistem Yönü (`systemStatus`) gerçek zamanlı dengeleme verileridir ve teslimat saati tamamlandıktan yaklaşık 2 saat sonra ilan edilir. $D-1$ saat 12:00 kesme anında kamuya açıklanmış en son saat muhafazakar olarak **09:00** ($t - 15 - h$) olarak tanımlanmıştır.
* **PTF Yayın Kuralı:** $D-1$ gününün 24 saatlik PTF değerleri, $D-2$ günü saat 14:00'te kesinleşmektedir. Dolayısıyla $D-1$ 12:00 kesme anında $D-1$'in 24 saatinin tamamı kamuya açıktır; bu sebeple `ptf_lag_24` özelliği meşrudur. Buna karşılık teslimat gününün ($D$) PTF'si $D-1$ saat 14:00'te açıklandığından, GÖP öncesi kesme anında model özelliklerinde yer alamaz.
* **`smf_clean_same_hour_7d_mean` Özelliği:** Bu özellik, teslimat saatinden 48, 72, 96, 120, 144 ve 168 saat önceki 6 geçmiş günün aynı saat ortalamasıdır. Kesme anında $D-1$ gününün ilgili saatleri henüz kesinleşmediğinden (özellikle akşam saatleri), sızıntıyı önlemek adına $D-1$ dahil edilmemiş ve en küçük gecikme 48 saat (2 tam gün önce) olarak tutulmuştur. İsimlendirmesi 7 gün çağrışımı yapsa da fiilen 6 gecikme barındırmaktadır.

### Tablo: 20 Model Özelliğinin Yayın Kuralı ve Gecikme Takvimi

| Sıra | Özellik Adı | Açıklama | Kaynak Veri | Kesme Anında Gecikme (Lag) | Sızıntı Riski / Kural Notu |
| :---: | :--- | :--- | :--- | :---: | :--- |
| 1-24 | `hour_00` - `hour_23` | Saatlik kukla değişkenler (One-Hot) | Takvim | Bilinen takvim | Sızıntı riski yok (deterministik). |
| 25 | `is_weekend` | Hafta sonu göstergesi (Cumartesi-Pazar: 1) | Takvim | Bilinen takvim | Sızıntı riski yok (deterministik). |
| 26-27| `quarter_sin`, `quarter_cos` | Yıllık/çeyreklik harmonik mevsimsellik | Takvim | Bilinen takvim | Sızıntı riski yok (deterministik). |
| 28 | `load_forecast_mw` | Gün öncesi toplam tüketim yük tahmini | EPİAŞ Yük Tahmini | $D-1$ 10:30 - 11:30 | Kesme anından önce yayımlanan resmi tahmin. |
| 29 | `forecast_wind_ratio` | Rüzgar tahmini / Toplam yük oranı | EPİAŞ Rüzgar/Yük | $D-1$ 10:30 - 11:30 | Oran olarak ölçeklendirilmiş üretim tahmini. |
| 30 | `ptf_lag_24` | Dünkü aynı saat PTF | EPİAŞ GÖP | 24 saat | $D-2$ 14:00'te açıklanır; kesme anında 100% meşru. |
| 31 | `ptf_lag_48` | İki gün önceki aynı saat PTF | EPİAŞ GÖP | 48 saat | $D-3$ 14:00'te açıklanır; 100% meşru. |
| 32 | `ptf_lag_168` | Geçen haftaki aynı saat PTF | EPİAŞ GÖP | 168 saat | 7 gün önce açıklanır; 100% meşru. |
| 33 | `latest_known_smf` | Kesme anında bilinen en son SMF | EPİAŞ DGP | $15 + h$ saat (D-1 09:00) | Hedef saate göre 15 ila 38 saat öncesidir. |
| 34 | `legit_same_hour_smf` | Meşru en son aynı saat SMF | EPİAŞ DGP | $h \le 9 \rightarrow 24$h; $h \ge 10 \rightarrow 48$h | Saat 09:00 sonrası için D-2 kullanılır. |
| 35 | `smf_lag_48` | 2 gün önceki teslimat saati SMF | EPİAŞ DGP | 48 saat | Kesme anında tamamen kesinleşmiştir. |
| 36 | `smf_lag_72` | 3 gün önceki teslimat saati SMF | EPİAŞ DGP | 72 saat | Kesme anında tamamen kesinleşmiştir. |
| 37 | `smf_lag_168` | 1 hafta önceki aynı saat SMF | EPİAŞ DGP | 168 saat | Kesme anında tamamen kesinleşmiştir. |
| 38 | `smf_median_4w` | Son 4 haftanın aynı saat medyan SMF | EPİAŞ DGP | Lags [168, 336, 504, 672] | Resmi referans baz modelin temel girdisi. |
| 39 | `smf_clean_same_hour_7d_mean` | 6 günün aynı saat ortalaması | EPİAŞ DGP | Lags [48, 72, 96, 120, 144, 168] | **Not: İsmi 7 gün çağrıştırsa da sızıntıyı önlemek için 6 gecikme içerir.** |
| 40 | `dir_open_ratio_4w` | Son 4 haftanın aynı saat Açık oranı | EPİAŞ DGP | Lags [168, 336, 504, 672] | Yön modelleri için iklimsel frekans bazı. |

### B. Gelecek Veri Zehirleme Testi (Poison Test)
Özellik matrisi üretim kodunun point-in-time güvenilirliğini test etmek amacıyla, rastgele seçilen test günlerinde teslimat gününden önceki kesme anından ($D-1$ saat 12:00) sonraki tüm gerçekleşen SMF ve yön verileri, ayrıca $D$ gününün tüm PTF verileri `NaN` ile zehirlenmiştir.
* Zehirli ham veri üzerinden özellik matrisi baştan çıkarılmıştır.
* Hedef $D$ gününün 24 saatlik özellik matrisi incelendiğinde **hiçbir özellik sütununda `NaN` değer oluşmadığı** ve zehirsiz orijinal özellik matrisi ile farkın **$0.0$** olduğu assert ile programatik olarak teyit edilmiştir.

---

## 2. DÖNEMLER ARASI MODEL HATA VE NMAE KARŞILAŞTIRMASI

Nihai modeller, test kümesi (01.10.2025 - 30.09.2026) öncesindeki **14.664 saatlik** birleşik dönem (Eğitim + Doğrulama: 29.01.2024 - 30.09.2025) üzerinde eğitilmiştir. Aşağıdaki tablo, modellerin eğitim, doğrulama ve dış test dönemlerindeki seviyeden arındırılmış normalize mutlak hatalarını (NMAE) özetlemektedir.

### Tablo 8: Dönemler Arası Hata ve NMAE Dağılımı (`donemler_arasi_hata_ve_nmae_tablosu.csv`)

| Dönem | Saat Sayısı | Ortalama SMF (TL/MWh) | Ridge MAE (TL/MWh) | Ridge NMAE (%) | LightGBM MAE (TL/MWh) | LightGBM NMAE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Eğitim (Train)** | 5.904 | 2.153,21 | 498,46 | %23,15 | 363,05 | %16,86 |
| **Doğrulama (Val)** | 8.760 | 2.421,62 | 670,91 | %27,70 | 635,71 | %26,25 |
| **Dış Test (Test)** | 8.760 | 2.198,25 | 918,13 | %41,77 | 947,81 | %43,12 |

> **Gözlem:** LightGBM modeli eğitim kümesinde Ridge'e kıyasla belirgin şekilde daha düşük hata üretmiştir (363,05 TL vs 498,46 TL; NMAE %16,86 vs %23,15). Ancak doğrulama kümesinde bu fark daralmış (635,71 TL vs 670,91 TL), dış test kümesinde ise LightGBM'in hatası Ridge'in üzerine çıkmıştır (947,81 TL vs 918,13 TL). Bu durum, ağaç tabanlı karmaşık modellerin dağılım kaymalarında ve fiyat rejimi değişimlerinde doğrusal düzenlileştirilmiş modellere (Ridge) göre daha yüksek genelleme bozulması yaşayabildiğini göstermektedir.

---

## 3. FİYAT TAHMİNİ KARŞILAŞTIRMA SONUÇLARI

Diebold-Mariano (DM) testleri, saatlik serilerdeki otokorelasyon ve heteroskedastisiteyi arındırmak için **günlük ortalama kayıplar (N = 365 gün)** üzerinden **Newey-West (HAC, 7 lag)** standart hatalarıyla hesaplanmıştır. Kayıp farkı yönü $d_t = L(\text{Model}) - L(\text{Baz})$ şeklinde tanımlanmış olup **negatif değerler modelin daha düşük hataya sahip olduğunu** ifade etmektedir. Güven aralıkları 7 günlük blok bootstrap (1.000 replikasyon) ile üretilmiştir.

### Tablo 1: Fiyat Tahmini Modelleri Karşılaştırma Tablosu (`tablo1_fiyat_tahmini_karsilastirma.csv`)

| Model Adı | Model Ailesi | MAE (TL/MWh) | RMSE (TL/MWh) | NMAE (%) | sMAPE (%) | Resmi Baza Göre MAE Farkı (TL) | 95% Blok Bootstrap GA (TL) | Günlük HAC DM $t$-stat | Günlük HAC DM İki Taraflı $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Resmi Referans Baz: smf_median_4w** | Meşru Baz Model | **911,41** | **1.316,31** | **%41,46** | **%67,44** | **0,00** | - | 0,000 | - |
| **2. Haftalık Aynı Saat Bazı (Lag 168)** | Uzak Baz Model | 1.011,80 | 1.497,20 | %46,03 | %71,34 | +100,39 | [+55,74, +150,43] | +3,978 | $p < 0.0001$ |
| **3. Dünkü Aynı Saat PTF (ptf_lag_24)** | Dünkü PTF Bazı | 865,57 | 1.246,04 | %39,38 | %67,91 | -45,84 | [-94,32, +0,21] | -1,930 | 0,0536 |
| **4. Meşru Dinamik Aynı Saat Bazı** | Dinamik Baz Model | 1.065,39 | 1.557,38 | %48,47 | %74,04 | +153,99 | [+102,38, +208,52] | +5,793 | $p < 0.0001$ |
| **5. GÖP Sonrası Doğal Kural: SMF = PTF** | GÖP Sonrası Doğal Baz | 682,75 | 1.019,49 | %31,06 | %59,78 | -228,66 | [-273,15, -179,57] | -9,608 | $p < 0.0001$ |
| **6. GÖP Öncesi: Ridge (Sabit Model)** | İstatistiksel | 918,13 | 1.104,21 | %41,77 | %68,32 | +6,72 | [-34,87, +52,61] | +0,291 | 0,7709 |
| **7. GÖP Öncesi: Ridge (Genişleyen Pencere)** | İstat. (Aylık Refit) | **891,97** | **1.095,85** | **%40,58** | **%68,73** | **-19,44** | **[-57,15, +22,18]** | **-0,940** | **0,3474** |
| **8. GÖP Öncesi: Ridge Ablasyonu (Yük/Rüzgarsız)**| İstat. Ablasyon | 899,33 | 1.114,42 | %40,91 | %69,61 | -12,07 | [-45,60, +24,17] | -0,649 | 0,5161 |
| **9. GÖP Öncesi: LightGBM (Sabit Model)** | Makine Öğrenmesi | 947,81 | 1.168,67 | %43,12 | %68,68 | +36,40 | [-17,43, +91,76] | +1,256 | 0,2093 |
| **10. GÖP Öncesi: LightGBM (Genişleyen Pencere)** | Makine Öğr. (Aylık Refit) | 897,82 | 1.128,03 | %40,84 | %69,03 | -13,59 | [-53,30, +29,84] | -0,631 | 0,5280 |
| **11. GÖP Sonrası: LightGBM Yayılım Modeli** | Makine Öğrenmesi | 730,14 | 1.003,11 | %33,21 | %68,27 | -181,27 | [-226,45, -131,63] | -7,425 | $p < 0.0001$ |

### Tablo 2: Özel Model Çiftleri İçin İki Yönlü Newey-West (HAC) Diebold-Mariano Testleri (`tablo2_ozel_ciftler_diebold_mariano.csv`)

| Karşılaştırma Çifti (Model 1 vs Model 2) | Model 1 MAE | Model 2 MAE | MAE Farkı (M1 - M2) | DM $t$ (MAE) | DM İki Taraflı $p$ (MAE) | Model 1 RMSE | Model 2 RMSE | RMSE Farkı (M1 - M2) | DM $t$ (RMSE) | DM İki Taraflı $p$ (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge (Sabit) vs Resmi Baz (smf_median_4w)** | 918,13 | 911,41 | +6,72 | +0,291 | 0,7709 | 1.104,21 | 1.316,31 | **-212,10** | -8,440 | **$p < 0.0001$** |
| **Ridge (Genişleyen) vs Resmi Baz (smf_median_4w)**| 891,97 | 911,41 | -19,44 | -0,940 | 0,3474 | 1.095,85 | 1.316,31 | **-220,47** | -8,931 | **$p < 0.0001$** |
| **Ridge (Sabit) vs Lag 168** | 918,13 | 1.011,80 | -93,67 | -3,438 | 0,0006 | 1.104,21 | 1.497,20 | **-392,99** | -10,669 | **$p < 0.0001$** |
| **Ridge (Genişleyen) vs ptf_lag_24** | 891,97 | 865,57 | +26,40 | +1,326 | 0,1849 | 1.095,85 | 1.246,04 | **-150,19** | -7,555 | **$p < 0.0001$** |
| **Ridge (Genişleyen) vs LightGBM (Genişleyen)** | 891,97 | 897,82 | -5,85 | -0,487 | 0,6266 | 1.095,85 | 1.128,03 | **-32,18** | -2,705 | **0,0068** |
| **LightGBM Yayılım vs SMF = PTF** | 730,14 | 682,75 | **+47,39** | +7,321 | **$p < 0.0001$** | 1.003,11 | 1.019,49 | **-16,38** | -1,810 | 0,0703 |

> **Temel İstatistiksel Çıkarımlar:**
> 1. **MAE Metriğinde Anlamlı Kazanç Yoktur:** GÖP öncesi kesme anında ne Sabit Ridge ($p = 0.7709$), ne Genişleyen Ridge ($p = 0.3474$), ne de Genişleyen LightGBM ($p = 0.5280$) resmi referans baz modele (`smf_median_4w`) karşı mutlak hata açısından istatistiki olarak anlamlı bir üstünlük sağlayamamaktadır. Benzer şekilde `ptf_lag_24` bazına karşı da MAE farkı anlamsızdır ($p = 0.1849$).
> 2. **RMSE Metriğinde Güçlü Kazanç Vardır:** Buna karşın karesel kayıp (RMSE) açısından modeller aşırı uç fiyat şoklarını naif medyan bazına kıyasla çok daha iyi törpülemektedir. Genişleyen Ridge modeli, resmi baza kıyasla karesel hatada 220,47 TL/MWh ($p < 0.0001$), `ptf_lag_24` bazına kıyasla ise 150,19 TL/MWh ($p < 0.0001$) hata azalışı sağlamaktadır.
> 3. **GÖP Sonrası Yayılım Modeli Karşıtlığı:** PTF açıklandıktan sonra uygulanan $SMF = PTF$ basit kuralı MAE metriğinde LightGBM yayılım modelinden 47,39 TL daha başarılıdır ($p < 0.0001$). LightGBM ise RMSE metriğinde 16,38 TL daha düşük hata vermektedir ($p = 0.0703$).

---

## 4. AYLIK VE SAATLİK HATA DAĞILIMI VE BAZ KARŞILAŞTIRMASI

Aşağıdaki tablolar, modellerin ve baz modellerin 12 ay ve 24 saat üzerindeki ayrışmasını göstermektedir. Ridge Sabit, Ridge Genişleyen, LightGBM Genişleyen, Resmi Baz (`smf_median_4w`) ve Dünkü PTF (`ptf_lag_24`) birlikte sunulmuştur.

### Tablo 3: 12 Aylık Hata ve Ortalama Hata (ME/Bias) Dağılımı (`tablo3_aylik_hata_ve_bias.csv`)

| Ay | Gözlem | Ortalama SMF (TL) | Ridge Sabit ME/Bias (TL) | Ridge Sabit MAE (TL) | Ridge Genişleyen MAE (TL) | Resmi Baz (Medyan 4w) MAE (TL) | Dünkü PTF (ptf_lag_24) MAE (TL) | LightGBM Genişleyen MAE (TL) | Ridge(Gen.) - Resmi Baz Farkı (TL) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2025-10** | 744 | 2.599,90 | -92,53 | 683,04 | 683,04 | 633,32 | 641,39 | 655,26 | +49,72 |
| **2025-11** | 720 | 2.765,46 | -271,05 | 719,38 | 706,93 | 580,64 | 616,11 | 587,09 | +126,29 |
| **2025-12** | 744 | 2.911,50 | -208,67 | 626,22 | 596,78 | 507,69 | 489,23 | 564,92 | +89,09 |
| **2026-01** | 744 | 2.760,64 | -141,94 | 652,07 | 632,75 | 597,45 | 645,15 | 646,95 | +35,30 |
| **2026-02** | 672 | 1.850,99 | +352,62 | 1.016,24 | 1.019,81 | 1.150,53 | 1.142,40 | 1.012,90 | **-130,72** |
| **2026-03** | 744 | 1.687,62 | +182,99 | 1.017,02 | 1.007,46 | 1.216,53 | 1.028,53 | 1.031,16 | **-209,07** |
| **2026-04** | 720 | 1.185,86 | +403,24 | 1.186,13 | 1.127,50 | 1.148,25 | 921,44 | 1.255,67 | **-20,74** |
| **2026-05** | 744 | 875,30 | +389,76 | 1.141,77 | 1.026,39 | 938,18 | 818,64 | 1.007,57 | +88,21 |
| **2026-06** | 720 | 1.761,27 | -81,19 | 1.234,57 | 1.215,43 | 1.197,26 | 1.220,57 | 1.160,77 | +18,17 |
| **2026-07** | 744 | 2.515,33 | +124,87 | 945,97 | 923,07 | 1.129,39 | 934,47 | 1.003,71 | **-206,32** |
| **2026-08** | 744 | 2.905,90 | -112,69 | 805,86 | 795,62 | 836,10 | 904,02 | 889,61 | **-40,48** |
| **2026-09** | 720 | 2.507,16 | +95,76 | 1.014,28 | 996,65 | 1.034,85 | 1.063,21 | 982,07 | **-38,21** |
| **Ağırlıklı Ort.**| **8.760**| **2.198,25**| **+62,55** | **918,13** | **891,97** | **911,41** | **865,57** | **897,82** | **-19,44** |

> **Aylık Dağılım İncelemesi:**
> * Ridge (Genişleyen Pencere) modeli, Resmi Baza (`smf_median_4w`) karşı 12 ayın **6'sında** daha düşük MAE vermiştir (Şubat, Mart, Nisan, Temmuz, Ağustos, Eylül).
> * Dünkü PTF (`ptf_lag_24`) bazına karşı ise 12 ayın **7'sinde** daha düşük MAE elde etmiştir (Ocak, Şubat, Mart, Haziran, Temmuz, Ağustos, Eylül).
> * **Mayıs 2026 Ayrışması:** Mayıs ayında hidrolojik bolluk ve güneş üretimi nedeniyle ortalama SMF 875,30 TL'ye düşmüştür. Model geçmiş yüksek fiyatlı eğitim verisinin etkisiyle +389,76 TL pozitif bias (aşırı tahmin) üretmiş, MAE 1.026,39 TL seviyesinde kalmıştır.

### Tablo 4: 24 Saatin Tamamı İçin Saatlik MAE Dağılımı (`tablo4_saatlik_24_saat_mae.csv`)

| Saat | Ortalama SMF (TL) | Ridge Sabit MAE (TL) | Ridge Genişleyen MAE (TL) | Resmi Baz (Medyan 4w) MAE (TL) | Dünkü PTF (ptf_lag_24) MAE (TL) | Ridge(Gen.) - Resmi Baz Farkı (TL) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00:00** | 2.352,56 | 923,58 | 908,14 | 1.037,41 | 922,50 | **-129,27** |
| **01:00** | 2.302,59 | 894,43 | 873,61 | 978,64 | 856,39 | **-105,03** |
| **02:00** | 2.028,64 | 891,55 | 870,15 | 991,58 | 850,00 | **-121,43** |
| **03:00** | 1.836,01 | 874,18 | 845,50 | 972,79 | 825,13 | **-127,29** |
| **04:00** | 1.794,72 | 896,61 | 861,21 | 978,75 | 867,51 | **-117,54** |
| **05:00** | 1.920,68 | 864,94 | 831,72 | 959,38 | 853,89 | **-127,66** |
| **06:00** | 2.042,68 | 799,37 | 754,09 | 778,09 | 715,67 | **-24,00** |
| **07:00** | 1.864,43 | 809,51 | 754,33 | 652,91 | 620,21 | +101,42 |
| **08:00** | 2.145,56 | 857,26 | 826,81 | 738,17 | 868,01 | +88,63 |
| **09:00** | 2.043,45 | 971,27 | 942,66 | 856,78 | 1.009,97 | +85,88 |
| **10:00** | 1.667,67 | 1.009,76 | 979,29 | 952,11 | 986,34 | +27,18 |
| **11:00** | 1.702,85 | 1.043,44 | 1015,59 | 1.003,01 | 1.033,00 | +12,59 |
| **12:00** | 1.220,07 | 994,89 | 961,97 | 948,37 | 868,85 | +13,60 |
| **13:00** | 1.461,48 | 1.032,72 | 999,88 | 988,78 | 951,18 | +11,10 |
| **14:00** | 1.889,06 | 1.070,54 | 1041,88 | 1.054,13 | 1.039,31 | **-12,25** |
| **15:00** | 1.838,04 | 1.052,92 | 1024,05 | 1.003,77 | 980,36 | +20,29 |
| **16:00** | 2.199,21 | 979,64 | 947,18 | 916,34 | 858,25 | +30,84 |
| **17:00** | 2.648,47 | 875,94 | 855,40 | 785,86 | 778,04 | +69,54 |
| **18:00** | 2.930,63 | 829,20 | 814,46 | 806,87 | 789,68 | +7,59 |
| **19:00** | 3.178,38 | 805,33 | 788,89 | 814,15 | 720,89 | **-25,26** |
| **20:00** | 3.242,12 | 796,10 | 778,56 | 781,48 | 713,87 | **-2,93** |
| **21:00** | 3.192,99 | 846,89 | 831,75 | 796,25 | 751,94 | +35,50 |
| **22:00** | 2.783,14 | 932,10 | 922,98 | 977,98 | 932,88 | **-55,00** |
| **23:00** | 2.472,60 | 982,86 | 977,13 | 1.100,16 | 979,70 | **-123,03** |
| **Aritmetik Ort.**| **2.198,25**| **918,13** | **891,97** | **911,41** | **865,57** | **-19,44** |

> **Saatlik Dağılım İncelemesi:**
> * Ridge (Genişleyen) modeli, Resmi Baza karşı 24 saatin tam **12'sinde** daha iyi sonuç vermiştir.
> * Gece saatlerinde (00:00 - 06:00, 22:00 - 23:00) model resmi baza kıyasla 100-130 TL arasında belirgin bir MAE iyileşmesi sağlamaktadır.
> * Gündüz güneş üretimi ve öğle dip saatlerinde (07:00 - 13:00) ise fiyat oynaklığı ve fotovoltaik üretim nedeniyle medyan baz modelinin hatası Ridge'den daha düşüktür.
> * **Assert 1 Teyidi:** Tablo 1'deki genel Ridge Sabit MAE (**918,13 TL**), Tablo 3'teki aylık ağırlıklı Ridge Sabit MAE (**918,13 TL**) ve Tablo 4'teki 24 saatlik aritmetik ortalama Ridge Sabit MAE (**918,13 TL**) birbiriyle kuruşu kuruşuna örtüşmektedir.

---

## 5. BİAS, MINCER-ZARNOWITZ REGRESYONU VE KORELASYON ANALİZİ

### Tablo 11: Mincer-Zarnowitz Regresyonu ve Wald Testi Sonuçları (`mincer_zarnowitz_analizi.csv`)
Tahminlerin yansızlığını ve etkinlik derecesini test etmek amacıyla $SMF_t = \alpha + \beta \cdot \widehat{SMF}_t + \varepsilon_t$ Mincer-Zarnowitz regresyonu Newey-West (HAC, 7 lag) standart hataları ile tahmin edilmiştir ($N = 8.760$ saat).

| Model Adı | Gözlem ($N$) | $\alpha$ Sabit (HAC se) | $\beta$ Eğim (HAC se) | Belirlilik ($R^2$) | Wald Testi $H_0: (\alpha=0, \beta=1)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regresyon (Sabit)** | 8.760 | -530,51 (61,19) | 1,2131 (0,0229) | 0,4395 | $F = 86,87$ ($p = 1,37 \times 10^{-19}$) |
| **Resmi Baz (smf_median_4w)** | 8.760 | +812,48 (53,28) | 0,6164 (0,0181) | 0,2998 | $F = 480,54$ ($p = 4,48 \times 10^{-105}$) |

> **İstatistiksel Yorum:**
> 1. Her iki model için de $H_0: \alpha=0, \beta=1$ hipotezi kuvvetle reddedilmektedir ($p < 0.0001$).
> 2. Ridge modelinde $\beta = 1,2131 > 1$ bulunması, L2 ceza teriminin (shrinkage) tahminleri ortalamaya doğru büzdüğünü ve aşırı uç değerleri bastırdığını **göstermektedir**.
> 3. Resmi baz modelde ise $\beta = 0,6164 < 1$ ve $\alpha = +812,48$ bulunmuştur; bu durum naif medyanın cari piyasa fiyat değişimlerine yeterince hızlı uyum sağlayamadığını ve düşük fiyat dönemlerinde yukarı yönlü atalet taşıdığını göstermektedir.

### Ek Tablo: Aylık Ortalama Hata (ME) ile Piyasa Fiyatı Korelasyonu (`aylik_me_smf_korelasyon.csv`)

| Gözlem Sayısı ($n$) | Pearson Korelasyon Katsayısı ($r$) | $p$-değeri | 95% Güven Aralığı (Fisher $z$) |
| :---: | :---: | :---: | :---: |
| 12 ay | **-0,8355** | **0,0007** | **[-0,95, -0,50]** |

> **Bias Mekanizması Açıklaması:** Aylık gerçekleşen ortalama SMF ile Ridge modelinin ortalama hatası (ME / Bias) arasında $r = -0,8355$ düzeyinde istatistiki olarak anlamlı negatif bir korelasyon bulunmaktadır ($p = 0.0007$, %95 GA: $[-0,95, -0,50]$). Bu durum, modelin piyasa fiyat seviyesinin düştüğü dönemlerde (örneğin Mayıs 2026'da) pozitif bias (fiyatı olduğundan yüksek tahmin etme), fiyatların yükseldiği kış aylarında ise negatif bias (eksik tahmin etme) ürettiğini göstermektedir.

---

## 6. SİSTEM YÖNÜ SINIFLANDIRMA PERFORMANSI

Yön tahmini iki sınıflı (Açık: 1 vs Fazla/Dengede: 0) olarak kurgulanmıştır.
* **Açık Oranı Kayması:** Sistem Açık oranı, Eğitim+Doğrulama döneminde (2024 - 2025 Q3) **%64,38** iken dış test döneminde **%55,74**'e gerilemiştir.
* **İki Sınıflı Tercihin Gerekçesi:** 3 sınıflı sistemde "Dengede" durumu piyasa verisinde yalnızca **%0,44** sıklıkla görülmektedir (8.760 test saatinin sadece 39'u). Dengeleme Güç Piyasası'nda çift fiyatlandırma ve ceza mekanizması doğrudan sistemin yönüne (Açık veya Fazla) bağlı olduğundan, pratik piyasa riski açısından iki sınıflı kurgu temel alınmıştır.
* **Birincil ve İkincil İstatistiksel Testler:** Zaman serilerinde ardışık saatlerin bağımlılığı nedeniyle McNemar testi tip 1 hata riskini artırabilmektedir. Bu sebeple **7 günlük blok bootstrap ile %95 güven aralığı birincil test** olarak benimsenmiş; McNemar testi ise ikincil karşılaştırma olarak bağımlılık uyarısıyla raporlanmıştır.

### Tablo 5: Yön Sınıflandırma Modelleri Karşılaştırma Tablosu (`tablo5_yon_siniflandirma_karsilastirma.csv`)

| Model Adı | Doğruluk (Acc) | Baza Göre Acc GA | Dengeli Doğruluk | MCC | Baza Göre MCC GA | F1 Skoru | ROC AUC | Brier Skoru | BSS vs Hafta/Saat (%) | BSS vs 4w Açık (%) | McNemar Testi (İkincil) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Resmi Referans Baz: Hafta İçi/Sonu x Saat Çoğunluk** | **0,6046** | - | **0,5692** | **0,1774** | - | 0,7120 | 0,6357 | 0,2404 | %0,00 | %9,10 | Referans Baz |
| **2. Saat Bazlı Çoğunluk Bazı (Eğitimden)** | 0,5671 | [-5,05, -2,19]% | 0,5147 | 0,0730 | [-0,129, -0,076] | 0,7144 | - | - | - | - | $\chi^2 = 85,5$ ($p < 0.0001$) |
| **3. Son 4 Haftanın Aynı Saat Açık Oranı Bazı** | 0,6089 | [-1,64, +2,72]% | 0,5850 | 0,1876 | [-0,033, +0,055] | 0,6934 | 0,6392 | 0,2644 | %-10,01 | %0,00 | $\chi^2 = 0,6$ ($p = 0.4488$) |
| **4. Genel Çoğunluk Sınıfı (Her Zaman Açık)** | 0,5574 | [-6,45, -2,71]% | 0,5000 | 0,0000 | [-0,209, -0,145] | 0,7158 | - | - | - | - | $\chi^2 = 105,1$ ($p < 0.0001$) |
| **5. Lojistik Regresyon (Validasyon Eşikli)** | **0,6312** | **[-0,14, +5,79]%** | **0,6203** | **0,2449** | **[+0,016, +0,122]** | 0,6836 | **0,6668** | **0,2303** | **%4,17** | **%12,89** | $\chi^2 = 21,2$ ($p < 0.0001$) |
| **6. LightGBM Sınıflandırıcı (Val Eşikli)** | **0,6279** | **[-0,70, +5,50]%** | **0,6256** | **0,2501** | **[+0,026, +0,118]** | 0,6591 | **0,6812** | **0,2261** | **%5,92** | **%14,48** | $\chi^2 = 13,2$ ($p = 0.0003$) |

### Tablo 6: Üç Sınıflı Model Sonuçları (`tablo6_uc_sinifli_yon_karsilastirma.csv`)

| Model Adı | Doğruluk (Acc) | Dengeli Doğruluk | Macro F1 | MCC |
| :--- | :---: | :---: | :---: | :---: |
| **1. Saat Bazlı Çoğunluk Bazı (3 Sınıflı)** | 0,5667 | 0,3436 | 0,2742 | 0,0734 |
| **2. Hafta İçi/Sonu x Saat Bazı (3 Sınıflı)** | 0,5904 | 0,3677 | 0,3334 | 0,1487 |
| **3. LightGBM 3 Sınıflı Model** | **0,6281** | **0,4087** | **0,4028** | **0,2386** |

### Tablo 7: 24 Saat İçin Model Doğruluğu vs. Eğitimden Türetilen Saat Bazı (`tablo7_saatlik_yon_dogruluk.csv`)

| Saat | Test Açık Oranı (%) | Eğitimden Türetilen Saat Bazı Doğruluğu (%) | LightGBM Doğruluğu (%) | Model - Eğitim Bazı Farkı (% Puan) | Oracle (Test Verisi Çoğunluk Oranı) (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **00:00** | 54,52 | 54,52 | 55,62 | +1,10 | 54,52 |
| **01:00** | 60,82 | 60,82 | 59,73 | -1,10 | 60,82 |
| **02:00** | 50,41 | 50,41 | 52,60 | +2,19 | 50,41 |
| **03:00** | 42,47 | 42,47 | 56,99 | **+14,52** | 57,53 |
| **04:00** | 40,00 | 40,00 | 59,45 | **+19,45** | 60,00 |
| **05:00** | 46,85 | 46,85 | 53,15 | +6,30 | 53,15 |
| **06:00** | 54,52 | 54,52 | 58,63 | +4,11 | 54,52 |
| **07:00** | 52,05 | 52,05 | 64,66 | +12,60 | 52,05 |
| **08:00** | 58,90 | 58,90 | 71,23 | +12,33 | 58,90 |
| **09:00** | 57,81 | 57,81 | 72,60 | +14,79 | 57,81 |
| **10:00** | 47,40 | 47,40 | 65,75 | **+18,36** | 52,60 |
| **11:00** | 48,22 | 48,22 | 60,00 | +11,78 | 51,78 |
| **12:00** | 38,36 | 61,64 | 63,56 | +1,92 | 61,64 |
| **13:00** | 44,38 | 44,38 | 60,55 | **+16,16** | 55,62 |
| **14:00** | 56,44 | 56,44 | 61,10 | +4,66 | 56,44 |
| **15:00** | 51,78 | 51,78 | 63,84 | +12,05 | 51,78 |
| **16:00** | 60,55 | 60,55 | 65,75 | +5,21 | 60,55 |
| **17:00** | 67,95 | 67,95 | 73,42 | +5,48 | 67,95 |
| **18:00** | 73,97 | 73,97 | 75,07 | +1,10 | 73,97 |
| **19:00** | 71,51 | 71,51 | 69,32 | -2,19 | 71,51 |
| **20:00** | 71,78 | 71,78 | 68,22 | -3,56 | 71,78 |
| **21:00** | 70,14 | 70,14 | 67,67 | -2,47 | 70,14 |
| **22:00** | 60,00 | 60,00 | 55,07 | -4,93 | 60,00 |
| **23:00** | 56,99 | 56,99 | 52,88 | -4,11 | 56,99 |
| **Ortalama** | **55,74** | **56,71** | **62,79** | **+6,08** | **59,27** |

> **Assert 2 Teyidi:** Tablo 5'teki Saat Bazlı Çoğunluk Bazı doğruluğu (%56,71) ile Tablo 7'deki Eğitim Saat Bazı ortalaması (%56,71); ve Tablo 5'teki LightGBM doğruluğu (%62,79) ile Tablo 7'deki LightGBM ortalaması (%62,79) birebir aynıdır. Test dönemi Oracle çoğunluk oranı ise %59,27 ile bağımsız bir sütun olarak doğrulanmıştır.

---

## 7. ÇOKLU DOĞRUSAL BAĞLANTI (VIF) VE KATSAYI İŞARET KARARLILIĞI

### Tablo 9: Sürekli Değişkenler İçin Çoklu Doğrusal Bağlantı (VIF) Analizi (`vif_analiz_tablosu.csv`)

| Özellik Adı | VIF (Tüm Değişkenler Dahil) | VIF (Residual Load Çıkarılmış - Model Sürümü) | Durum / Açıklama |
| :--- | :---: | :---: | :--- |
| `load_forecast_mw` | 3.559,86 | **22,76** | Residual load ile tam doğrusal bağımlıydı; arındırıldı. |
| `forecast_residual_load_mw` | 3.097,79 | **Çıkarıldı** | `load_forecast - wind_forecast` ilişkisi VIF patlaması yaratıyordu. |
| `forecast_wind_ratio` | 35,93 | **3,76** | Kabul edilebilir aralıkta. |
| `latest_known_smf` | 6,91 | **6,91** | Düşük çoklu bağlantı. |
| `ptf_lag_24` | 29,71 | **29,67** | Fiyat serileri arası korelasyon. |
| `ptf_lag_48` | 33,88 | **33,63** | Fiyat serileri arası korelasyon. |
| `ptf_lag_168` | 35,78 | **35,64** | Fiyat serileri arası korelasyon. |
| `legit_same_hour_smf` | 20,13 | **20,08** | Orta düzey çoklu bağlantı. |
| `smf_lag_48` | 25,26 | **25,23** | Orta düzey çoklu bağlantı. |
| `smf_lag_72` | 14,66 | **14,65** | Orta düzey çoklu bağlantı. |
| `smf_lag_168` | 20,98 | **20,89** | Orta düzey çoklu bağlantı. |
| `smf_median_4w` | 30,32 | **30,30** | Orta düzey çoklu bağlantı. |
| `smf_clean_same_hour_7d_mean`| 72,47 | **71,96** | Yüksek gecikmeli ortalama bağlantısı. |

### Ek Tablo: Ridge Katsayıları ve İşaret Kararlılığı Analizi (1.000 Blok Bootstrap) (`ridge_katsayi_ve_isaret_kararliligi.csv`)

Aşağıda, 7 günlük blok bootstrap (1.000 replikasyon) ile hesaplanan standartlaştırılmış Ridge katsayıları ($\beta$), %95 güven aralıkları ve katsayının pozitif çıkma sıklığı verilmiştir. **İktisadi yorumlar nedensel kesinlik değil, model atfı (model attribution) olarak değerlendirilmelidir.**

| Özellik | $\beta$ Katsayısı | Bootstrap Ortalama | Bootstrap 95% GA | Pozitif Oran (%) | İşaret Kararlılığı | İktisadi Yorum (Model Atfı) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `load_forecast_mw` | **+225,22** | +225,94 | [+169,87, +284,79] | **%100,0** | Kararlı (Pozitif) | Tüketim yükü arttıkça model daha yüksek fiyat atfetmektedir. |
| `smf_median_4w` | **+168,46** | +167,45 | [+117,65, +218,97] | **%100,0** | Kararlı (Pozitif) | 4 haftalık aynı saat medyan fiyat çıpası kuvvetli pozitif katkı sağlar. |
| `forecast_wind_ratio` | **-104,89** | -104,50 | [-143,52, -69,51] | **%0,0** | Kararlı (Negatif) | Rüzgar payı arttıkça model merit-order etkisiyle fiyatı düşürmektedir. |
| `hour_12` | **-89,54** | -90,38 | [-108,75, -72,06] | **%0,0** | Kararlı (Negatif) | Güneş üretimi etkisiyle günün en derin fiyat düşüş saati (öğle dip). |
| `ptf_lag_24` | **+93,43** | +91,14 | [+54,49, +127,69] | **%100,0** | Kararlı (Pozitif) | Dünkü PTF seviyesi doğrudan fiyat referansı oluşturmaktadır. |
| `ptf_lag_168` | **+79,91** | +82,09 | [+34,34, +127,31] | **%100,0** | Kararlı (Pozitif) | Haftalık aynı saat PTF çıpası. |
| `legit_same_hour_smf` | **+63,38** | +63,05 | [+38,19, +86,21] | **%100,0** | Kararlı (Pozitif) | Bilinen son aynı saat SMF değeri. |
| `latest_known_smf` | **+53,81** | +52,30 | [+16,78, +89,13] | **%99,8** | Kararlı (Pozitif) | Kesme anında açıklanmış en güncel SMF seviyesi. |
| `hour_13` | **-70,92** | -70,97 | [-87,11, -54,50] | **%0,0** | Kararlı (Negatif) | Öğle dip saatleri devamı. |
| `hour_11` | **-63,46** | -64,06 | [-77,71, -51,13] | **%0,0** | Kararlı (Negatif) | Öğle dip saatleri başlangıcı. |
| `hour_10` | **-59,08** | -59,48 | [-73,63, -45,33] | **%0,0** | Kararlı (Negatif) | Sabah sonrası fotovoltaik üretim artışı. |
| `is_weekend` | **-34,93** | -33,96 | [-69,55, +2,33] | **%3,3** | Kararlı (Negatif) | Hafta sonu sanayi tüketim düşüşü. |

---

## 8. ÇOKLU KESME ANI İLE GERÇEK UFUK ANALİZİ

Teslimat saati ile tahmin ufku arasındaki karışıklığı (confounding) gidermek amacıyla, dış test dönemindeki tüm 365 günün sabit akşam puant saatleri (18:00 - 21:00 arası 4 saat, toplam 1.460 gözlem) üç farklı kesme anında test edilmiştir.

### Tablo 10: Akşam Puant Saatleri (18:00 - 21:00) İçin Kesme Anına Göre Başarım (`tablo10_gercek_ufuk_analizi.csv`)

| Kesme Anı (Cutoff) | Gerçek Ufuk | Kullanılan Model | Ridge MAE (TL/MWh) | Resmi Baz (Medyan 4w) MAE (TL/MWh) | Dünkü PTF (ptf_lag_24) MAE (TL/MWh) | Son Bilinen SMF Baz MAE (TL/MWh) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **$D-1$ saat 12:00 (GÖP Öncesi)** | 30 - 33 Saat Öncesi | Ridge (Sabit) | **819,38** | 799,69 | 744,10 | 1.519,95 |
| **$D$ günü saat 06:00 (Gün İçi Sabah)** | 12 - 15 Saat Öncesi | Ridge (Sabit) | **808,42** | 799,69 | 744,10 | 1.519,86 |
| **$D$ günü saat 15:00 (Teslimat Öncesi)**| 3 - 6 Saat Öncesi | Ridge (Sabit) | **818,38** | 799,69 | 744,10 | 1.832,81 |

> **Assert 3 Teyidi ve Bulguların Yorumu:**
> 1. **Assert Doğrulaması:** Tablo 4'teki 18:00, 19:00, 20:00 ve 21:00 saatlerinin Ridge Sabit MAE ortalaması $(829,20 + 805,33 + 796,10 + 846,89) / 4 = \mathbf{819,38\text{ TL}}$ olup, Tablo 10'daki D-1 12:00 kesme anı puant MAE'si ile (**819,38 TL**) kuruşu kuruşuna örtüşmektedir.
> 2. **Ufuk Dinamiği:** Kesme anı teslimat saatine yaklaştığında (30 saatten 3 saate indiğinde) Ridge modelinin puant hatası neredeyse sabit kalmaktadır (819 TL $\rightarrow$ 808 TL $\rightarrow$ 818 TL). Buna karşılık "Son Bilinen SMF" naif bazının hatası 1.519 TL - 1.832 TL gibi son derece yüksek seviyelerdedir. Modelin puant saatlerindeki başarısı, son bilinen anlık fiyattan ziyade saat kuklaları, yük tahmini ve PTF çıpasına dayanmaktadır.

---

## 9. KOD VE TABLO EŞLEŞTİRME DOĞRULAMASI

Revizyon V5 kapsamındaki tüm sayısal değerler, tek bir master tahmin dosyasından (`data/processed/test_predictions_master.parquet`) ve tek bir çalışma kimliği (`run_1792585978`) altında üretilmiştir. Raporda yer alan tüm tablolar ve kaynak CSV dosyalarının boyut ve SHA-256 sağlama toplamı (checksum) eşleştirmesi aşağıda sunulmuştur.

### Tablo 12: Rapor Tabloları, Kaynak Dosyalar ve SHA-256 Sağlama Toplamı Eşleştirmesi (`tablo_eslestirme_ve_dogrulama.csv`)

| Rapor Tablosu | Kaynak Dosya Yolu | Boyut (Bayt) | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| **Tablo 1: Fiyat Tahmini Karşılaştırma** | `reports/tables/tablo1_fiyat_tahmini_karsilastirma.csv` | 1.830 | `d8c032987308a5c1a09ac087b6269b2a850fbc56dfed46e87048c1a309d29cbc` |
| **Tablo 2: Özel Çiftler İçin HAC DM Testleri** | `reports/tables/tablo2_ozel_ciftler_diebold_mariano.csv` | 977 | `491096638f6fec29c976b7f56abcf85c48d7ff961f923786eea53630fa6f051f` |
| **Tablo 3: 12 Aylık Hata ve ME Dağılımı** | `reports/tables/tablo3_aylik_hata_ve_bias.csv` | 1.273 | `2b43dbbfca30e67f5af3f010176b704b96cd2e109aa05f12bb709e56f2c0f384` |
| **Tablo 4: 24 Saatlik MAE Dağılımı** | `reports/tables/tablo4_saatlik_24_saat_mae.csv` | 1.717 | `7a77027ae2fe8d8d9030eefbe454c9df5d54f227f5170533f5b67c2c6e1cb270` |
| **Tablo 5: Yön Sınıflandırma Karşılaştırma** | `reports/tables/tablo5_yon_siniflandirma_karsilastirma.csv` | 1.149 | `d4936a8859cfd1e3f5c7601e57f9770f4fde3211eb786b0d1ffd15f1d154c2cb` |
| **Tablo 6: 3 Sınıflı Yön Modeli ve Bazları** | `reports/tables/tablo6_uc_sinifli_yon_karsilastirma.csv` | 315 | `57fbacc0cc4b97bbc451d20ed47d3ef8134b6b88c1555d03593576fa9617c4ca` |
| **Tablo 7: 24 Saat Yön Doğruluk vs Saat Bazı**| `reports/tables/tablo7_saatlik_yon_dogruluk.csv` | 1.384 | `39e2ec162b3ee87d97988e1e849a84b0c577084b3bcabfdd093c2d270e046ae1` |
| **Tablo 8: Dönemler Arası Hata ve NMAE** | `reports/tables/donemler_arasi_hata_ve_nmae_tablosu.csv` | 324 | `82e3b335a4a890471d84f5fd2d2843bf194b8f014c9197628f8f0dc34e1e12a5` |
| **Tablo 9: Çoklu Doğrusal Bağlantı (VIF)** | `reports/tables/vif_analiz_tablosu.csv` | 690 | `5927b7c350fd084eb3798764faa1209f1e69f46c4c87d9a82f59b8594af06790` |
| **Tablo 10: Gerçek Ufuk Analizi (18-21)** | `reports/tables/tablo10_gercek_ufuk_analizi.csv` | 537 | `0deb8993bd96c9765c37853050a0509eccd84c8dc3a530605e4b0792fade07c0` |
| **Tablo 11: Mincer-Zarnowitz Regresyonu** | `reports/tables/mincer_zarnowitz_analizi.csv` | 316 | `5ddaf88ff3fad188c363ad8133d1f818c57bc553dca6815ff19f33ef030e5514` |
| **Ek Tablo: Aylık ME-SMF Korelasyonu** | `reports/tables/aylik_me_smf_korelasyon.csv` | 102 | `8d0c429140157e8b3d1ec7c33f72913e8918de9faf5d9e2e7137018b808eb0de` |
| **Ek Tablo: Ridge Katsayı Kararlılığı** | `reports/tables/ridge_katsayi_ve_isaret_kararliligi.csv` | 3.319 | `c035536f8cb01657b17b371685b976c508aa682ad98f11cb4ff4167e5bcc8f15` |
| **Kaynak Veri: EPİAŞ Resmi Piyasa Verisi** | `data/raw/epias_official_market_data_full.parquet` | 806.446 | `e4748553a38b4fff3f013b75b1b473190f35059972c984f9e4192b4eef941dd5` |
| **Master Tahmin: Tek Kaynak Parquet** | `data/processed/test_predictions_master.parquet` | 955.464 | `b67783dedff62b1bc8c0712d8d2f7a95b498323018a29304523f12d2b530f211` |

---

## 10. DANIŞMAN HOCAYA SUNULACAK KARAR SORUSU

> **Danışman Hocaya Soru:**  
> *"Hocam, tezimizin ana problem kurgusunu **GÖP Öncesi (D-1 saat 12:00)** mi, yoksa **GÖP Sonrası (D-1 saat 14:00 - PTF açıklandıktan sonra)** mı konumlandırmalıyız?  
> * **GÖP Öncesi Analiz:** MAE metriğinde resmi referans baza (`smf_median_4w`, 911,41 TL) ve `ptf_lag_24` bazına (865,57 TL) karşı modellerimiz istatistiki olarak anlamlı bir fark yaratamamaktadır (Ridge Genişleyen MAE: 891,97 TL, $p = 0.3474$). Buna karşılık karesel hata metriğinde (RMSE) aşırı uç fiyat şoklarını törpüleyerek resmi baza karşı **220,47 TL/MWh ($p < 0.0001$)** düzeyinde güçlü ve anlamlı bir hata azalışı sağlamaktadır.  
> * **GÖP Sonrası Analiz:** PTF kesinleştikten sonra uygulanan basit **$SMF = PTF$ kuralı (682,75 TL MAE)**, yayılımı modelleyen makine öğrenmesi modelinden (730,14 TL MAE) mutlak hata açısından belirgin biçimde daha başarılıdır ($p < 0.0001$). LightGBM ise sadece RMSE metriğinde 16,38 TL'lik sınırlı bir fark sunabilmektedir ($p = 0.0703$).  
> Bu bulgular ışığında, piyasa yapıcı bir katılımcının karar anı açısından çalışmayı GÖP öncesine odaklamak mı daha uygundur, yoksa her iki aşamanın bilgi değerini karşılaştırmalı olarak sunmak mı tercih edilmelidir?"*
