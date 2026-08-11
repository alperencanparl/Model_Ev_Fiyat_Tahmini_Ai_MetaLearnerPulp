import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
import pulp
import warnings
import plotly.express as px

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# SAYFA AYARLARI
# ---------------------------------------------------------
st.set_page_config(page_title="Yatırım Motoru", page_icon="📈", layout="wide")
st.title("🏡 Akıllı Gayrimenkul Yatırım & Optimizasyon Portföyü")
st.caption("Meta-Learner (Stacking AI) + Yöneylem Araştırması (PuLP Linear Programming)")

# ---------------------------------------------------------
# 1. VERİ SİMÜLASYONU (Mevcut 'df' Yapınız)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'Ev_Oda_sayisi': np.random.randint(1, 6, n),
        'Ev_Banyo_sayisi': np.random.randint(1, 3, n),
        'Lokasyon': np.random.choice(['İstanbul', 'Bursa', 'Antalya'], n),
        'Semt_Sinifi': np.random.choice(['A', 'B', 'C'], n),
        'Ev_tipi': np.random.choice(['Daire', 'Villa', 'Müstakil'], n),
        'Ev_Fiyati': np.random.randint(1_000_000, 5_000_000, n)
    })
    return df

df = load_data()

# ---------------------------------------------------------
# 2. META-LEARNER EĞİTİMİ
# ---------------------------------------------------------
@st.cache_resource
def train_meta_model(data):
    X = data.drop(columns=["Ev_Fiyati"])
    y = data["Ev_Fiyati"]
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    base_models = [
        ('lightgbm', lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)),
        ('random_forest', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('grad_boost', GradientBoostingRegressor(n_estimators=50, random_state=42))
    ]
    
    meta_model = StackingRegressor(
        estimators=base_models,
        final_estimator=lgb.LGBMRegressor(n_estimators=30, max_depth=3, random_state=42, verbose=-1),
        passthrough=True,
        cv=3
    )
    meta_model.fit(X_encoded, y)
    return meta_model, X_encoded.columns

with st.spinner("Yapay Zeka Komitesi (Meta-Learner) Eğitiliyor..."):
    meta_model, encoded_columns = train_meta_model(df)

# ---------------------------------------------------------
# 3. YAN PANEL (SIDEBAR) - BÜTÇE AYARLARI
# ---------------------------------------------------------
st.sidebar.header("🕹️ Yatırım Parametreleri")
yatirim_butcesi = st.sidebar.number_input(
    "Toplam Yatırım Bütçesi (TL):", 
    min_value=500_000, 
    max_value=50_000_000, 
    value=5_000_000, 
    step=500_000
)

incelenecek_ilan_sayisi = st.sidebar.slider("Piyasada Taranacak İlan Sayısı:", 5, 50, 20)

# ---------------------------------------------------------
# 4. OPTİMİZASYON HESAPLAMA
# ---------------------------------------------------------
aday_portfoy = df.head(incelenecek_ilan_sayisi).copy().reset_index(drop=True)

# Piyasa İlan Fiyatı Simülasyonu
np.random.seed(42)
aday_portfoy["Ilan_Fiyati"] = aday_portfoy["Ev_Fiyati"] * np.random.uniform(0.80, 1.15, size=len(aday_portfoy))

# Meta-Learner Tahmini
X_portfoy = aday_portfoy.drop(columns=["Ev_Fiyati", "Ilan_Fiyati"], errors="ignore")
X_portfoy_encoded = pd.get_dummies(X_portfoy, drop_first=True).reindex(columns=encoded_columns, fill_value=0)

aday_portfoy["Meta_Tahmin"] = meta_model.predict(X_portfoy_encoded)
aday_portfoy["Potansiyel_Kar"] = aday_portfoy["Meta_Tahmin"] - aday_portfoy["Ilan_Fiyati"]

# PuLP Optimizasyonu
prob = pulp.LpProblem("Gayrimenkul_Optimizasyonu", pulp.LpMaximize)
ev_secimleri = pulp.LpVariable.dicts("Ev", aday_portfoy.index, cat="Binary")

prob += pulp.lpSum([ev_secimleri[i] * aday_portfoy.loc[i, "Potansiyel_Kar"] for i in aday_portfoy.index])
prob += pulp.lpSum([ev_secimleri[i] * aday_portfoy.loc[i, "Ilan_Fiyati"] for i in aday_portfoy.index]) <= yatirim_butcesi

prob.solve(pulp.PULP_CBC_CMD(msg=0))

# Seçilen Evler
secilen_indeksler = [i for i in aday_portfoy.index if ev_secimleri[i].varValue == 1]
secilen_evler = aday_portfoy.loc[secilen_indeksler]

toplam_harcama = secilen_evler["Ilan_Fiyati"].sum()
toplam_kar = secilen_evler["Potansiyel_Kar"].sum()

# ---------------------------------------------------------
# 5. EKRAN GÖSTERİMİ (METRİKLER & TABLOLAR)
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tanımlanan Bütçe", f"{yatirim_butcesi:,.0f} TL")
col2.metric("Kullanılan Sermaye", f"{toplam_harcama:,.0f} TL")
col3.metric("Kalan Nakit", f"{yatirim_butcesi - toplam_harcama:,.0f} TL")
col4.metric("Maksimum Tahmini Kâr", f"{toplam_kar:,.0f} TL", delta=f"%{(toplam_kar/toplam_harcama)*100:.1f} Getiri" if toplam_harcama > 0 else "0%")

st.markdown("---")
st.subheader("🎯 Optimizasyon Tarafından Seçilen En Kârlı Ev Portföyü")

if len(secilen_evler) > 0:
    # 1. TABLO GÖSTERİMİ
    ekran_tablosu = secilen_evler[[
        'Ev_tipi', 'Lokasyon', 'Semt_Sinifi', 'Ev_Oda_sayisi', 'Ev_Banyo_sayisi', 
        'Ilan_Fiyati', 'Meta_Tahmin', 'Potansiyel_Kar'
    ]].copy()
    
    ekran_tablosu.columns = [
        'Tip', 'Lokasyon', 'Semt', 'Oda', 'Banyo', 
        'İlan Fiyatı (TL)', 'AI Değeri (TL)', 'Tahmini Kâr (TL)'
    ]
    
    st.dataframe(
        ekran_tablosu.style.format({
            'İlan Fiyatı (TL)': '{:,.0f}',
            'AI Değeri (TL)': '{:,.0f}',
            'Tahmini Kâr (TL)': '{:,.0f}'
        }),
        use_container_width=True
    )

    # 2. GRAFİK GÖSTERİMİ (BUNLAR DA 'if' İÇİNDE İÇERİDEN BAŞLAMALI)
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📊 Ev Başına Potansiyel Kâr Dağılımı")
        fig_kar = px.bar(
            secilen_evler,
            x=secilen_evler.index.astype(str),
            y="Potansiyel_Kar",
            color="Ev_tipi",
            hover_data=["Lokasyon", "Semt_Sinifi", "Ev_Oda_sayisi"],
            labels={"x": "Ev ID", "Potansiyel_Kar": "Tahmini Kâr (TL)", "Ev_tipi": "Ev Tipi"},
            text_auto=',.0f'
        )
        fig_kar.update_layout(showlegend=True, xaxis_title="Seçilen Ev ID")
        st.plotly_chart(fig_kar, use_container_width=True)

    with col_chart2:
        st.subheader("⚖️ İlan Fiyatı vs. AI Değeri")
        
        karsilastirma_df = secilen_evler.reset_index().melt(
            id_vars=["index", "Ev_tipi"], 
            value_vars=["Ilan_Fiyati", "Meta_Tahmin"],
            var_name="Metrik", 
            value_name="Tutar"
        )
        karsilastirma_df["Metrik"] = karsilastirma_df["Metrik"].map({
            "Ilan_Fiyati": "İlan Fiyatı", 
            "Meta_Tahmin": "AI Tahmini"
        })

        fig_karsilastirma = px.bar(
            karsilastirma_df,
            x="index",
            y="Tutar",
            color="Metrik",
            barmode="group",
            labels={"index": "Ev ID", "Tutar": "Tutar (TL)"}
        )
        fig_karsilastirma.update_layout(xaxis_title="Seçilen Ev ID")
        st.plotly_chart(fig_karsilastirma, use_container_width=True)

else:
    st.warning("Bu bütçe ile kârlı bir yatırım fırsatı bulunamadı.")



