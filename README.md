# Model_Ev_Fiyat_Tahmini_Ai_MetaLearnerPulp
# 🏡 Gayrimenkul Değerleme ve Portföy Optimizasyonu Platformu

Bu proje, emlak ilanlarının adil piyasa değerlerini yapay zeka ile tahmin eden ve belirlenen yatırım bütçesine göre en kârlı ev kombinasyonunu seçen kişisel bir **Karar Destek Sistemidir**.

---

## 🛠️ Sistem Mimarisi ve Mantığı

### 1. Fiyat Tahmin Modeli (Stacking Ensemble)
* **Birinci Katman (Temel Modeller):** `LightGBM`, `Random Forest` ve `Gradient Boosting` algoritmaları ilan verileri üzerinde eğitilir.
* **İkinci Katman (Meta-Learner):** Bu üç modelin tahminlerini birleştiren ana `LightGBM` modeli, evin nihai adil piyasa değerini hesaplar.

### 2. Portföy Seçimi (PuLP Optimizasyonu)
* Yapay zekanın tahmin ettiği değerden ilan fiyatı çıkarılarak her evin potansiyel kârı bulunur:
  `Potansiyel Kâr = AI Tahmin Değeri - İlan Fiyatı`
* **PuLP** kütüphanesi kullanılarak, tanımlanan bütçe sınırını aşmayacak şekilde toplam kârı en yüksek seviyeye çıkaran ev kombinasyonu otomatik olarak seçilir.

### 3. Kullanıcı Arayüzü (Streamlit & Plotly)
* **Streamlit** arayüzü sayesinde bütçe miktarı ve incelenecek ilan sayısı panelden ayarlanabilir.
* Seçilen en ideal evler; kullanılan sermaye, beklenen kâr ve karşılaştırmalı grafiklerle ekranda gösterilir.

---

## 📁 Dosya Yapısı

* `app.py`: Streamlit arayüzünü ve optimizasyon mantığını çalıştıran ana dosya.
* `Model_Ev_Fiyat_Tahmini_Ai.py`: Veri işleme, model eğitimi ve tahmin süreçlerini içeren dosya.

---

## 💻 Kurulum ve Çalıştırma

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install streamlit pandas numpy lightgbm scikit-learn pulp plotly
