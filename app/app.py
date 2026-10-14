"""
App estimasi harga mobil bekas (v2).
Jalankan dari folder project:   streamlit run app/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PATH_MODEL = Path(__file__).resolve().parent.parent / "models" / "model_v2.joblib"
MODEL_LAIN = "Model lain / tidak ada di daftar"
MEREK_LAIN = "Merek lain (BMW, Nissan, Hyundai, dll)"

st.set_page_config(page_title="Estimasi Harga Mobil Bekas", page_icon="🚗", layout="centered")


@st.cache_resource
def muat_model():
    return joblib.load(PATH_MODEL)


def rp(x):
    return f"Rp {x / 1e6:,.0f} juta".replace(",", ".")


def prediksi(paket, merek, model, tahun, km, mesin_l, bahan_bakar, bodi, transmisi):
    info = paket["info"]
    lengkap = f"{merek} {model}".strip()
    n = info["jumlah_per_model"].get(lengkap, 0)
    model_g = lengkap if lengkap in info["model_dikenal"] else f"{merek} LAINNYA"
    keyakinan = "Tinggi" if n >= 30 else "Sedang" if n >= info["min_model"] else "Rendah"
    baris = pd.DataFrame([{
        "merek": merek, "model_g": model_g, "bahan_bakar": bahan_bakar, "bodi": bodi,
        "transmisi": transmisi, "umur": max(info["tahun_acuan"] - tahun, 0),
        "km": km, "mesin_l": 0.0 if bahan_bakar == "Listrik" else mesin_l}])
    est = float(paket["est"].predict(baris)[0])
    f_low, f_high = info["faktor_rentang"][keyakinan]
    return {"estimasi": est, "rendah": est * f_low, "tinggi": est * f_high,
            "keyakinan": keyakinan, "n": n}


def cek_input(tahun, km, mesin_l, bahan_bakar, tahun_acuan):
    peringatan = []
    umur = max(tahun_acuan - tahun, 0)
    if km is not None and not np.isnan(km):
        if umur >= 2 and km < 3000:
            peringatan.append("Kilometer sangat kecil untuk mobil seumur ini. Pastikan angkanya benar.")
        if km / max(umur, 1) > 40000:
            peringatan.append("Kilometer per tahun sangat tinggi (di atas 40.000 km/tahun). Pastikan angkanya benar.")
    if bahan_bakar != "Listrik" and mesin_l <= 0:
        peringatan.append("Kapasitas mesin 0 liter hanya masuk akal untuk mobil listrik.")
    return peringatan


try:
    paket = muat_model()
except Exception as e:  # file model belum ada / versi library tidak cocok
    st.error("Model belum bisa dimuat. Jalankan dulu notebook 02_zenodo_model.ipynb sampai selesai.")
    st.caption(f"Detail teknis: {e}")
    st.stop()

info = paket["info"]
pilihan = info["pilihan"]
daftar_merek = list(info["merek_didukung"])

st.title("🚗 Estimasi Harga Mobil Bekas")
st.caption("Isi spesifikasi mobil, dapatkan estimasi harga, rentang harga, dan tingkat keyakinannya. "
           "Dibuat dari ±18.800 iklan OLX & Mobil123 (Jabodetabek, 2025–2026).")

# ---------- input spesifikasi (dipakai kedua tab) ----------
st.subheader("Spesifikasi mobil")
c1, c2 = st.columns(2)
with c1:
    merek = st.selectbox("Merek", daftar_merek + [MEREK_LAIN], key="merek")
if merek == MEREK_LAIN:
    st.warning(
        f"Merek ini belum didukung, jadi tidak ada angka yang kuberikan. Datanya terlalu sedikit untuk "
        f"menebak harga dengan jujur. Merek yang didukung saat ini: {', '.join(daftar_merek)}."
    )
    st.stop()

with c2:
    model_pilihan = st.selectbox("Model", info["model_per_merek"][merek] + [MODEL_LAIN], key="model")
model = "" if model_pilihan == MODEL_LAIN else model_pilihan

c3, c4 = st.columns(2)
with c3:
    tahun = st.number_input("Tahun mobil", min_value=1985, max_value=info["tahun_acuan"], value=2018, step=1)
with c4:
    transmisi = st.selectbox("Transmisi", pilihan["transmisi"])

c5, c6 = st.columns(2)
with c5:
    bahan_bakar = st.selectbox("Bahan bakar", pilihan["bahan_bakar"])
with c6:
    bodi = st.selectbox("Tipe bodi", [b for b in pilihan["bodi"] if b != "Tidak diketahui"])

c7, c8 = st.columns(2)
with c7:
    km_tidak_tahu = st.checkbox("Kilometer tidak diketahui")
    km_input = st.number_input("Kilometer (km)", min_value=0, max_value=600000, value=50000,
                               step=1000, disabled=km_tidak_tahu)
    km = np.nan if km_tidak_tahu else float(km_input)
with c8:
    listrik = bahan_bakar == "Listrik"
    mesin_input = st.number_input("Kapasitas mesin (liter)", min_value=0.0, max_value=7.0, value=0.0 if listrik else 1.5,
                                  step=0.1, disabled=listrik, help="Mobil listrik otomatis 0.")
    mesin_l = 0.0 if listrik else float(mesin_input)

for pesan in cek_input(int(tahun), km, mesin_l, bahan_bakar, info["tahun_acuan"]):
    st.warning(pesan)
if km_tidak_tahu:
    st.info("Tanpa kilometer, akurasi estimasi turun. Isi kilometer kalau kamu tahu.")

hasil = prediksi(paket, merek, model, int(tahun), km, mesin_l, bahan_bakar, bodi, transmisi)

PENJELASAN = {
    "Tinggi": "Model ini punya banyak data ({n} iklan), jadi estimasinya paling bisa dipercaya.",
    "Sedang": "Data model ini lumayan ({n} iklan), tapi belum banyak. Rentang harga bisa lebih lebar.",
    "Rendah": "Model ini jarang atau tidak ada di data, jadi ini estimasi kasar berdasarkan mobil sejenis. "
              "Anggap sebagai gambaran awal, bukan patokan harga.",
}


def tampilkan_hasil():
    st.metric("Estimasi harga", rp(hasil["estimasi"]))
    st.write(f"**Rentang wajar:** {rp(hasil['rendah'])} sampai {rp(hasil['tinggi'])}")
    badge = {"Tinggi": "✅", "Sedang": "🟡", "Rendah": "⚠️"}[hasil["keyakinan"]]
    st.write(f"**Tingkat keyakinan:** {badge} {hasil['keyakinan']}")
    st.caption(PENJELASAN[hasil["keyakinan"]].format(n=hasil["n"]))


tab_jual, tab_beli = st.tabs(["Saya penjual", "Saya pembeli"])

with tab_jual:
    st.markdown("#### Berapa harga pasang yang pantas?")
    tampilkan_hasil()
    st.caption("Rentang dihitung dari seberapa jauh tebakan model meleset pada data uji, dan mencakup sekitar "
               "4 dari 5 harga iklan mobil serupa. Pasang di dekat batas atas kalau ingin ruang tawar, "
               "atau di dekat estimasi kalau ingin cepat laku.")

with tab_beli:
    st.markdown("#### Apakah harga yang dipasang wajar?")
    harga_pasang = st.number_input("Harga yang dipasang penjual (juta Rp)", min_value=10, max_value=20000,
                                   value=int(round(hasil["estimasi"] / 1e6)), step=5)
    harga = harga_pasang * 1e6
    selisih = (harga - hasil["estimasi"]) / hasil["estimasi"] * 100
    if harga < hasil["rendah"]:
        st.success(f"Di bawah rentang wajar ({selisih:+.0f}% dari estimasi). Bisa jadi murah, atau ada kondisi "
                   "yang perlu dicek (surat, tabrak, banjir, km turun).")
    elif harga > hasil["tinggi"]:
        st.error(f"Di atas rentang wajar ({selisih:+.0f}% dari estimasi). Layak ditawar, atau cek apakah "
                 "ada kelebihan khusus.")
    else:
        st.info(f"Masih dalam rentang wajar ({selisih:+.0f}% dari estimasi).")
    tampilkan_hasil()
    st.caption("Ini bantuan pembanding, bukan pengganti cek fisik dan surat mobil.")

with st.expander("Seberapa akurat estimasi ini?"):
    st.write("Rata-rata meleset dari harga iklan sebenarnya, diuji dengan validasi silang pada data yang tidak "
             "dipakai melatih:")
    st.dataframe(pd.DataFrame(info["skor_cv"]).T.round(1))
    st.write("**Model dikenal** artinya model mobil itu ada di data latih. **Model baru** artinya seluruh model "
             "itu sengaja disembunyikan dari data latih. Makin sedikit data sebuah model, makin besar meleset-nya.")

with st.expander("Batasan & sumber data"):
    st.markdown(
        "- Harga yang dipakai adalah **harga iklan**, bukan harga transaksi akhir.\n"
        "- Data hanya dari Jabodetabek dan merek tertentu; merek lain belum didukung.\n"
        "- Kondisi fisik, riwayat servis, dan kelengkapan surat tidak ikut dihitung.\n"
        "- Sumber data: Thania, I. & Oktavia, T. (2026), Zenodo, DOI 10.5281/zenodo.21792616, CC BY 4.0."
    )
