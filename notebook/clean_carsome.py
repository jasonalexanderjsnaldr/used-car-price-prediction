"""
Memuat dataset Carsome (Kaggle: "Used Car Listings in Indonesia", CC0, data dari carsome.id 2023)
dan mengubahnya ke format yang sama dengan data Zenodo yang sudah bersih.

Pakai di notebook:

    from clean_zenodo import bersihkan
    from clean_carsome import muat_carsome
    zen, laporan = bersihkan("../data/dataset_raw_29747.csv")
    carsome = muat_carsome("../data/used_car.csv", zen)
"""
import re

import numpy as np
import pandas as pd

# Tipe bodi untuk model yang tidak ada di data Zenodo.
BODI_MANUAL = {
    ("Wuling", "Almaz"): "SUV", ("Wuling", "Almaz RS"): "SUV",
    ("Wuling", "Confero"): "MPV", ("Wuling", "Confero S"): "MPV",
    ("Wuling", "Cortez"): "MPV", ("Wuling", "Cortez S"): "MPV",
    ("Nissan", "Grand Livina"): "MPV", ("Nissan", "Livina"): "MPV",
    ("Nissan", "Serena"): "MPV", ("Nissan", "March"): "Hatchback",
    ("Nissan", "X-Trail"): "SUV", ("Nissan", "Juke"): "SUV",
    ("Mazda", "2"): "Hatchback", ("Mazda", "CX-5"): "SUV",
    ("Mazda", "CX-3"): "SUV", ("Mazda", "Biante"): "MPV",
    ("Mitsubishi", "Eclipse Cross"): "SUV", ("Mitsubishi", "Colt L300"): "Pickup",
}


def _modus(df, kunci, kolom):
    return (
        df.dropna(subset=[kolom]).groupby(kunci)[kolom]
        .agg(lambda s: s.value_counts().index[0]).to_dict()
    )


def _mesin(nama):
    m = re.findall(r"(?<![\d.])([1-6]\.\d)(?!\d)", nama)
    return float(m[-1]) if m else np.nan


def muat_carsome(path, zen):
    """Kembalikan DataFrame Carsome dengan kolom yang sama seperti data Zenodo bersih (sebagian)."""
    c = pd.read_csv(path)
    c = c.rename(columns={
        "car name": "nama", "brand": "merek", "year": "tahun", "mileage (km)": "km_ribu",
        "transmission": "transmisi", "price (Rp)": "harga",
    })
    c["nama_u"] = c["nama"].str.upper().str.strip()

    kamus = {m: {str(x).upper(): str(x) for x in g["model"].unique()} for m, g in zen.groupby("merek")}

    def cari_model(merek, nama):
        k = kamus.get(merek, {})
        cocok = [u for u in k if nama == u or nama.startswith(u + " ")]
        if cocok:
            return k[max(cocok, key=len)]
        t = nama.split()
        if t[0] == "GRAND" and len(t) > 1:
            return " ".join(t[:2]).title()
        return t[0] if t[0].isdigit() or "-" in t[0] else t[0].title()

    c["model"] = [cari_model(m, n) for m, n in zip(c["merek"], c["nama_u"])]
    c["model"] = c["model"].replace({"Cx-3": "CX-3", "Cx-5": "CX-5", "X-trail": "X-Trail"})

    c["mesin_l"] = c["nama_u"].map(_mesin)
    c["km"] = (c["km_ribu"] * 1000).round()          # di file asli km ditulis dalam ribuan (10.508 = 10.508 km)

    # bahan bakar & bodi tidak ada di Carsome: isi dari model yang sama di Zenodo
    bb_3 = _modus(zen, ["merek", "model", "mesin_l"], "bahan_bakar")
    bb_2 = _modus(zen, ["merek", "model"], "bahan_bakar")
    bodi_2 = _modus(zen[zen["bodi"] != "Tidak diketahui"], ["merek", "model"], "bodi")
    c["bahan_bakar"] = [
        bb_3.get((m, mo, ml), bb_2.get((m, mo), "Bensin"))
        for m, mo, ml in zip(c["merek"], c["model"], c["mesin_l"])
    ]
    c["bodi"] = [
        bodi_2.get((m, mo), BODI_MANUAL.get((m, mo), "Tidak diketahui"))
        for m, mo in zip(c["merek"], c["model"])
    ]
    c["sumber"] = "Carsome"

    kolom = ["sumber", "merek", "model", "tahun", "km", "transmisi",
             "bahan_bakar", "mesin_l", "bodi", "harga"]
    return c[kolom].reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from clean_zenodo import bersihkan

    zen, _ = bersihkan(sys.argv[1])
    car = muat_carsome(sys.argv[2], zen)
    print(car.shape)
    print(car["merek"].value_counts())
    print(car.head())
