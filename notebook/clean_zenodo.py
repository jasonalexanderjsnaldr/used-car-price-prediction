"""
Pembersihan dataset mobil bekas Zenodo (dataset_raw_29747.csv).

Sumber data: Thania, I. (2026), Zenodo, DOI 10.5281/zenodo.21792616 (CC BY 4.0).
Pakai di notebook:

    from clean_zenodo import bersihkan
    df, laporan = bersihkan("../data/dataset_raw_29747.csv")
"""
import re

import numpy as np
import pandas as pd

MEREK_VALID = {
    "honda": "Honda", "toyota": "Toyota", "daihatsu": "Daihatsu", "suzuki": "Suzuki",
    "mitsubishi": "Mitsubishi", "wuling": "Wuling", "byd": "BYD",
    "mercedes-benz": "Mercedes-Benz", "mercedes benz": "Mercedes-Benz",
    "bmw": "BMW", "hyundai": "Hyundai", "mazda": "Mazda", "dfsk": "DFSK",
    "chery": "Chery", "denza": "Denza", "mini cooper": "MINI", "mini": "MINI",
    "smart": "Smart",
}

BODI_MAP = {
    "SUV": "SUV", "Jeep": "SUV", "MPV": "MPV", "Hatchback": "Hatchback",
    "Compact & City Car": "Hatchback", "Sedan": "Sedan", "Coupe": "Coupe",
    "Van": "Van", "Minibus": "Van", "Pickup": "Pickup",
}

MERC_FAMILY = re.compile(
    r"^(AMG GT|GLA|GLB|GLC|GLE|GLK|GLS|GL|CLA|CLS|SLK|SL|ML|A|B|C|E|G|S|V)(?=[\s\d\-]|$)"
)


def _angka_harga(x):
    if pd.isna(x):
        return np.nan
    s = re.sub(r"[^0-9]", "", str(x))
    return float(s) if s else np.nan


def _km_tengah(x):
    """'10 - 15K km' -> 12500, '50.000-55.000 km' -> 52500, '>300.000 km' -> 300000."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace("km", "").replace(" ", "").replace(">", "")
    pakai_k = "k" in s
    s = s.replace("k", "")
    nilai = []
    for bagian in s.split("-"):
        bagian = bagian.replace(".", "")
        if not bagian.isdigit():
            return np.nan
        v = float(bagian)
        nilai.append(v * 1000 if pakai_k else v)
    return float(np.mean(nilai)) if nilai else np.nan


def _mesin_liter(cc_raw, varian, judul):
    """cc eksak ('1498 cc') > angka di varian ('2.0 Bensin-AT') > angka di judul."""
    if isinstance(cc_raw, str):
        m = re.match(r"^\s*(\d{3,4})\s*cc\s*$", cc_raw)
        if m:
            return int(m.group(1)) / 1000.0
    for teks in (varian, judul):
        if isinstance(teks, str):
            m = re.search(r"(?<![\d.])([1-6]\.\d)(?![\d])", teks)
            if m:
                return float(m.group(1))
    return np.nan


def _keluarga_mercy(model):
    if not isinstance(model, str):
        return model
    m = MERC_FAMILY.match(model.upper().strip())
    return m.group(1) + "-Class" if m else model


def bersihkan(path):
    d = pd.read_csv(path)
    laporan = {"baris_awal": len(d)}

    # 1. buang baris kosong dan baris judul kolom yang ikut ter-scrape
    d = d[d["url"].notna()].copy()
    d = d[d["merek_raw"].astype(str).str.lower() != "merk"].copy()
    d = d[d["merek_raw"].notna()].copy()
    laporan["setelah_buang_kosong_dan_header"] = len(d)

    # 2. merek: samakan huruf, pulihkan merek Mercedes yang tertulis sebagai nama kelas
    kecil = d["merek_raw"].astype(str).str.strip().str.lower()
    merek = kecil.map(MEREK_VALID)
    mercy_kelas = kecil.str.contains(r"-class$", regex=True) | kecil.isin(["cls400", "cla200"])
    merek = merek.where(~mercy_kelas, "Mercedes-Benz")
    d["merek"] = merek
    # baris kolom bergeser / merek tak dikenal tidak bisa dipulihkan murah -> dibuang
    d = d[d["merek"].notna()].copy()

    # 3. harga, tahun
    d["harga"] = d["harga_raw"].map(_angka_harga)
    d["tahun_n"] = pd.to_numeric(d["tahun"], errors="coerce")
    d = d[(d["harga"] >= 20_000_000) & (d["tahun_n"] >= 1985) & (d["tahun_n"] <= 2026)].copy()
    d["tahun_n"] = d["tahun_n"].astype(int)
    laporan["setelah_harga_tahun_valid"] = len(d)

    # 4. model (samakan nama kelas Mercedes)
    model = d["model_raw"].astype(str).str.strip()
    model = model.where(d["model_raw"].notna(), np.nan)
    model = model.where(model.str.lower() != d["merek"].str.lower(), np.nan)
    model = model.where(~model.isin(["Lainnya", "Lain-lain", "Lain Lain"]), "Lainnya")
    d["model"] = np.where(d["merek"] == "Mercedes-Benz", model.map(_keluarga_mercy), model)
    d["model"] = d["model"].astype("object")
    d = d[d["model"].notna() & (d["model"] != "nan")].copy()

    # 5. kilometer (tengah rentang), transmisi, bahan bakar
    d["km"] = d["kilometer_raw"].map(_km_tengah)
    d.loc[d["km"] > 600_000, "km"] = np.nan
    d["transmisi"] = d["transmisi_raw"].where(d["transmisi_raw"].isin(["Automatic", "Manual"]))
    bb = d["bb_raw"].replace({"Petrol": "Bensin"})
    d["bahan_bakar"] = bb.where(bb.isin(["Bensin", "Diesel", "Hybrid", "Listrik"]))

    # 6. mesin (liter); mobil listrik = 0
    d["mesin_l"] = [
        _mesin_liter(c, v, j) for c, v, j in zip(d["cc_raw"], d["varian_raw"], d["judul"])
    ]
    d.loc[d["bahan_bakar"] == "Listrik", "mesin_l"] = 0.0

    # 7. tipe bodi: dari kolom, sisanya diisi dari modus bodi model yang sama
    d["bodi"] = d["bodi_raw"].map(BODI_MAP)
    modus = (
        d.dropna(subset=["bodi"]).groupby(["merek", "model"])["bodi"]
        .agg(lambda s: s.value_counts().index[0])
    )
    kunci = list(zip(d["merek"], d["model"]))
    isi = pd.Series([modus.get(k, np.nan) for k in kunci], index=d.index)
    d["bodi"] = d["bodi"].fillna(isi).fillna("Tidak diketahui")

    # 8. lokasi (kota) dan tanggal iklan
    d["kota"] = d["lokasi_raw"].astype(str).str.split(",").str[0].str.strip()
    d.loc[d["lokasi_raw"].isna(), "kota"] = np.nan
    t = d["tanggal_raw"].astype(str)
    tgl = pd.to_datetime(t, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    d["tgl"] = tgl.fillna(pd.to_datetime(t, format="%d-%m-%Y", errors="coerce"))
    d["bulan_idx"] = (d["tgl"].dt.year - 2025) * 12 + d["tgl"].dt.month

    # 9. duplikat: url yang sama, lalu iklan kembar (mobil sama yang dipasang di banyak kota)
    d = d.drop_duplicates("url").copy()
    laporan["setelah_hapus_url_ganda"] = len(d)
    tanda = ["merek", "model", "varian_raw", "tahun_n", "harga", "kilometer_raw", "warna_raw", "transmisi"]
    d = d.drop_duplicates(tanda).copy()
    laporan["setelah_hapus_iklan_kembar"] = len(d)

    kolom = [
        "sumber", "batch", "merek", "model", "varian_raw", "tahun_n", "km", "transmisi",
        "bahan_bakar", "mesin_l", "bodi", "kota", "tgl", "bulan_idx", "harga",
    ]
    d = d[kolom].rename(columns={"varian_raw": "varian", "tahun_n": "tahun"}).reset_index(drop=True)
    laporan["baris_akhir"] = len(d)
    return d, laporan


if __name__ == "__main__":
    import sys

    df, lap = bersihkan(sys.argv[1] if len(sys.argv) > 1 else "dataset_raw_29747.csv")
    for k, v in lap.items():
        print(f"{k}: {v}")
    print(df.head())
