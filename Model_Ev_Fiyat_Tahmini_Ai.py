import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
import shap 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor


np.random.seed(42)

tarihler = pd.date_range(start="2026-01-01", end="2026-12-31", freq="1h")
df = pd.DataFrame({"tarih": tarihler})



df["Lokasyon"] = np.random.choice(["Antalya", "Istanbul", "Bursa", "Izmir", "Ankara"], size=len(df))
df["Semt_Sinifi"] = np.random.choice(["A", "B", "C", "D"], size=len(df))
df["Semt_Kalabaligi"] = np.random.choice(["Yuksek", "Orta", "Dusuk"], size=len(df))
df["Ev_tipi"] = np.random.choice(["Daire", "Villa", "Müstakil Ev"], size=len(df))
df["Ev_Banyo_sayisi"] = np.random.randint(1, 4, size=len(df))
df["Ev_Oda_sayisi"] = np.random.randint(1, 6, size=len(df))

df["Ev_Fiyati"] = (df["Ev_Oda_sayisi"] * 50000 + df["Ev_Banyo_sayisi"] * 20000 + df["Ev_tipi"].map({"Daire": 10000, "Villa": 30000, "Müstakil Ev": 40000})).astype(int)

df.to_csv("emlak_verisi.csv", index=False)
print("Emlak verisi 'emlak_verisi.csv' dosyasına kaydedildi.")

#Feature
df["Ev_Oda_Carpim_Banyo"] = df["Ev_Oda_sayisi"] * df["Ev_Banyo_sayisi"]

#Anomali 
df.loc[df["Ev_Fiyati"] > 300000, "Anomali"] = 1
df.loc[df["Ev_Fiyati"] <= 300000, "Anomali"] = 0

df["Ev_Banyo_sayisi_Lag1"] = df["Ev_Banyo_sayisi"].shift(1)
df["Ev_Banyo_sayisi_Lag24"] = df["Ev_Banyo_sayisi"].shift(24)

df["Ev_Banyo_sayisi_Rolling_Mean_5"] = df["Ev_Banyo_sayisi"].rolling(window=5).mean()

feature_cols = [
    "Ev_Oda_sayisi",
    "Ev_tipi",
    "Lokasyon",
    "Semt_Kalabaligi",
    "Semt_Sinifi",
    "Ev_Banyo_sayisi",
    "Ev_Banyo_sayisi_Lag1",
    "Ev_Banyo_sayisi_Lag24",
    "Ev_Banyo_sayisi_Rolling_Mean_5",
]

X = df[feature_cols].copy()

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
categorical_cols = ["Lokasyon", "Semt_Sinifi", "Semt_Kalabaligi", "Ev_tipi"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X = X.fillna(0)

y = df["Ev_Fiyati"]

kategori_indeksleri = [feature_cols.index(col) for col in categorical_cols]

#K-Prototypes Kumeleme

from kmodes.kprototypes import KPrototypes

kproto = KPrototypes(n_clusters=3, random_state=42)
kumeler = kproto.fit_predict(X.to_numpy(), categorical=kategori_indeksleri)

df["Kume_Etiketi"] = kumeler

print(df[['Lokasyon', "Ev_Fiyati", 'Kume_Etiketi']].head(10))

import lightgbm as lgb

egitim_orani = int(0.8 * len(df))

X_train, X_test = X[:egitim_orani], X[egitim_orani:]
y_train, y_test = y[:egitim_orani], y[egitim_orani:]

model_lgb = lgb.LGBMRegressor(random_state=42)
model_lgb.fit(X_train, y_train)

y_pred = model_lgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

hatalar = y_test.values - y_pred
esik_degeri = hatalar.mean() + (3 * hatalar.std())
anomaliler = hatalar > esik_degeri

sonuc_tablosu = df.iloc[egitim_orani:].copy()
sonuc_tablosu["Modelin_Beklentisi"] = np.around(y_pred, 1)
sonuc_tablosu["Anomali Mi"] = anomaliler

print("---Model Basari Karnesi---")
print(f"Ortalama Sapma(MAE): {mae}")
print(f"Basari Skoru (R^2): {r2:.4f}\n")

print("Lokasyon: Antalya, Istanbul, Bursa, Izmir, Ankara")
print("Semt_Sinifi: A, B, C, D")
print("Semt_Kalabaligi: Yuksek, Orta, Dusuk ")
print("Ev_tipi: Daire, Villa, Müstakil Ev")
print("Ev_Banyo_sayisi: 1 ile 4 arasi")
print("Ev_Oda_sayisi: 1 ile 6 arasi")

lokasyon = input("Lokasyonu Seciniz: ")
semt_sinifi = input("Semt_Sinifini_Seciniz: ")
semt_kalabalik = input("Semt_Kalabaligini Seciniz: ")
Ev_Tipi = input("Ev_Tipini Seciniz: ")
Ev_Banyo_sayisi = int(input("Ev_Banyo_sayisini Seciniz: "))
Ev_Oda_sayisi = int(input("Ev_Oda_sayisini Seciniz: "))

yeni_veri = pd.DataFrame({
    "Lokasyon": [lokasyon],
    "Semt_Sinifi": [semt_sinifi],
    "Semt_Kalabaligi": [semt_kalabalik],
    "Ev_tipi": [Ev_Tipi],
    "Ev_Banyo_sayisi": [Ev_Banyo_sayisi],
    "Ev_Oda_sayisi": [Ev_Oda_sayisi],
    "Ev_Banyo_sayisi_Lag1": [0],
    "Ev_Banyo_sayisi_Lag24": [0],
    "Ev_Banyo_sayisi_Rolling_Mean_5": [0]
})

# Encode categorical columns in new data
for col in categorical_cols:
    if col in yeni_veri.columns:
        yeni_veri[col] = label_encoders[col].transform(yeni_veri[col])

# Ensure prediction input has same feature names and order as the training data
yeni_veri = yeni_veri.reindex(columns=X_train.columns, fill_value=0)

yeni_tahmin = model_lgb.predict(yeni_veri)

print(f"Modelin_Tahmini_Ev_Fiyati: {yeni_tahmin[0]:.2f} TL")

print("\nSHAP Anilizi Hesaplaniyor... Lutfen Bekleyin.")

#SHAP
explainer = shap.TreeExplainer(model_lgb)

shap_values = explainer(yeni_veri)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[0], show=False)

plt.title("Bu Evin Fiyati Neden Boyle Cikti", fontsize=14)
plt.tight_layout()
plt.show()

#Meta-Learner

from sklearn.linear_model import Ridge 

X_encoded = pd.get_dummies(X, drop_first = True)


# 1. Uzman Modeller
base_models = [
    ('lightgbm', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)),
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('grad_boost', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

#Meta_Learner

meta_model = StackingRegressor(
    estimators=base_models,
    # final_estimator=Ridge(alpha=1.0), lineer genel puana
    ## Düz çizgi çeken Ridge yerine, duruma göre karar veren ağaç tabanlı patron:
    final_estimator=lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=42)
    # Orijinal ev özelliklerini (Oda, Semt, Lokasyon) patron modele de göster!

    ,passthrough=True,

    cv=5 # 5 Katlamalı Çapraz Doğrulama (OOF) ile aşırı öğrenmeyi engeller

)

print("Meta-Learner Eğitiliyor...")
meta_model.fit(X_encoded, y)
print("Eğitim Tamamlandı!")

import pulp

# =========================================================
# YÖNEYLEM ARAŞTIRMASI (OPERATIONS RESEARCH) - YATIRIM OPTİMİZATÖRÜ
# =========================================================

# 1. PARAMETRE: Yatırımcının Toplam Bütçesi
YATIRIM_BUTCESI = 5_000_000  # 5 Milyon TL Bütçe

# 2. Piyasadan 20 Adet İlan Toplandığını Varsayalım (Simülasyon)
aday_portfoy = df.head(20).copy().reset_index(drop=True)

# İlanların gerçek fiyatları ve Meta-Learner tahminleri
X_portfoy = aday_portfoy.drop(columns=["tarih", "Ev_Fiyati"]).fillna(0)
X_portfoy_encoded = pd.get_dummies(X_portfoy, drop_first=True).reindex(columns=X_encoded.columns, fill_value=0)

aday_portfoy["Ilan_Fiyati"] = aday_portfoy["Ev_Fiyati"] * np.random.uniform(0.80, 1.15, size=len(aday_portfoy)) # İlan sahibinin istediği fiyat
aday_portfoy["Meta_Tahmin"] = meta_model.predict(X_portfoy_encoded) # Bizim modelin adil değer tahmini

# Beklenen Kâr = Modelin Adil Değer Tahmini - İlan Fiyatı
aday_portfoy["Potansiyel_Kar"] = aday_portfoy["Meta_Tahmin"] - aday_portfoy["Ilan_Fiyati"]

# ---------------------------------------------------------
# 3. YÖNEYLEM MODELİNİN (KNAPSACK / 0-1 TAM SAYI PROGRAMLAMA) KURULMASI
# ---------------------------------------------------------
# Amaç: Kârı Maksimum Yapmak (LpMaximize)
prob = pulp.LpProblem("Gayrimenkul_Portfoy_Optimizasyonu", pulp.LpMaximize)

# Karar Değişkeni: Her ev için Binary değişken (0 = Alma, 1 = Satın Al)
ev_secimleri = pulp.LpVariable.dicts("Ev", aday_portfoy.index, cat="Binary")

# AMAÇ FONKSİYONU: Seçilen evlerin toplam kârını maksimize et
prob += pulp.lpSum([ev_secimleri[i] * aday_portfoy.loc[i, "Potansiyel_Kar"] for i in aday_portfoy.index])

# KISIT 1: Seçilen evlerin toplam maliyeti Bütçeyi AŞAMAZ!
prob += pulp.lpSum([ev_secimleri[i] * aday_portfoy.loc[i, "Ilan_Fiyati"] for i in aday_portfoy.index]) <= YATIRIM_BUTCESI

# Modeli Çöz
prob.solve(pulp.PULP_CBC_CMD(msg=0))

# ---------------------------------------------------------
# 4. OPTİMİZASYON ROPORU
# ---------------------------------------------------------
secilen_indeksler = [i for i in aday_portfoy.index if ev_secimleri[i].varValue == 1]
secilen_evler = aday_portfoy.loc[secilen_indeksler]

toplam_harcama = secilen_evler["Ilan_Fiyati"].sum()
toplam_beklenen_kar = secilen_evler["Potansiyel_Kar"].sum()

print("\n" + "="*50)
print("     YÖNEYLEM ARAŞTIRMASI OPTİMİZASYON RAPORU     ")
print("="*50)
print(f"Tanımlanan Toplam Bütçe   : {YATIRIM_BUTCESI:,.2f} TL")
print(f"Kullanılan Toplam Sermaye : {toplam_harcama:,.2f} TL")
print(f"Kalan Nakit               : {YATIRIM_BUTCESI - toplam_harcama:,.2f} TL")
print(f"Maksimum Tahmini Kâr      : {toplam_beklenen_kar:,.2f} TL")
print(f"Satın Alınan Ev Sayısı    : {len(secilen_evler)} / 20")
print("-" * 50)

print("OPTIMAL SEÇİLEN EVLERİN LİSTESİ:")
for idx, row in secilen_evler.iterrows():
    print(f"-> Ev #{idx} | İlan Fiyatı: {row['Ilan_Fiyati']:,.0f} TL | Tahmini Kâr: {row['Potansiyel_Kar']:,.0f} TL")

print("OPTIMAL SEÇİLEN EVLER VE ÖZELLİKLERİ:\n")
for idx, row in secilen_evler.iterrows():
    # Kendi veri setindeki sütun isimlerine göre bu kısımları dinamik çekiyoruz:
    oda = row.get('Ev_Oda_sayisi', 'N/A')
    banyo = row.get('Ev_Banyo_sayisi', 'N/A')
    lokasyon = row.get('Lokasyon', 'N/A')
    semt = row.get('Semt_Sinifi', 'N/A')
    ev_tipi = row.get('Ev_tipi', 'N/A')
    
    print(f"-> Ev #{idx:<2} | {ev_tipi:<6} | Oda: {oda} | Banyo: {banyo} | Lokasyon: {lokasyon} | Semt: {semt}")
    print(f"      İlan Fiyati: {row['Ilan_Fiyati']:,.0f} TL | Model Degeri: {row['Meta_Tahmin']:,.0f} TL | Tahmini Kar: {row['Potansiyel_Kar']:,.0f} TL")
    print("-" * 80)
