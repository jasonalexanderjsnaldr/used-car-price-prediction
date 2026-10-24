"""
Estimasi Harga Mobil Bekas (v2.1)
Jalankan dari folder paling luar project:   streamlit run app/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ======================= ISI BAGIAN INI (boleh dikosongkan) =======================
NAMA_PEMBUAT = ""          # contoh: "Nama Kamu"
LINK_GITHUB = ""           # contoh: "https://github.com/usernamekamu/Used-car-price-prediction"
LINK_PORTOFOLIO = ""       # contoh: "https://namakamu.vercel.app"
# ==================================================================================

PATH_MODEL = Path(__file__).resolve().parent.parent / "models" / "model_v2.joblib"
MODEL_LAIN = "Model lain / tidak ada di daftar"
MEREK_LAIN = "Merek lain (BMW, Hyundai, Kia, dll)"

st.set_page_config(page_title="Estimasi Harga Mobil Bekas", page_icon="🚗", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--line:rgba(255,255,255,.09);--txt:#E9EDF7;--mut:#93A0BD;--dim:#6B7794;--cy:#22D3EE;--vi:#8B5CF6;--pk:#F472B6;
--ok:#34D399;--warn:#FBBF24;--bad:#FB7185;--org:#FB923C;--grad:linear-gradient(90deg,#22D3EE,#8B5CF6 55%,#F472B6)}
.stApp{font-family:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--txt);
background:radial-gradient(1000px 520px at 10% -8%,rgba(139,92,246,.24),transparent 62%),
radial-gradient(800px 460px at 100% 2%,rgba(34,211,238,.15),transparent 58%),#070B16}
.stApp p,.stApp label,.stApp button,.stApp input,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp table{font-family:'Plus Jakarta Sans',system-ui,sans-serif}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer,[data-testid="stDecoration"]{display:none!important}
.block-container{max-width:1160px;padding:1.4rem 1.2rem 3rem}

/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:28px;padding:2.2rem 2.2rem 1.8rem;margin:0 0 1.3rem;
background:linear-gradient(135deg,rgba(139,92,246,.17),rgba(34,211,238,.07) 55%,rgba(244,114,182,.10))}
.hero:before{content:"";position:absolute;top:-45%;right:-8%;width:520px;height:520px;border-radius:50%;
background:radial-gradient(circle,rgba(34,211,238,.24),transparent 65%);pointer-events:none}
.hero:after{content:"";position:absolute;inset:0;pointer-events:none;
background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);
background-size:34px 34px;-webkit-mask-image:radial-gradient(circle at 75% 15%,#000,transparent 70%);mask-image:radial-gradient(circle at 75% 15%,#000,transparent 70%)}
.hero>*{position:relative;z-index:1}
.badge{display:inline-flex;align-items:center;gap:8px;font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
color:#CFE8F6;background:rgba(255,255,255,.07);border:1px solid var(--line);padding:6px 13px;border-radius:999px}
.badge i{width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 4px rgba(52,211,153,.2);animation:pulse 2s infinite}
.hero h1{font-size:clamp(1.9rem,4.6vw,3.3rem)!important;line-height:1.08!important;font-weight:800!important;letter-spacing:-.03em!important;
margin:.95rem 0 .6rem!important;padding:0!important;color:#fff}
.grad{background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent}
.hero p.sub{max-width:660px;color:var(--mut);font-size:1.02rem;line-height:1.65;margin:0 0 1.4rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;max-width:720px}
.stat{padding:12px 16px;border-radius:16px;background:rgba(255,255,255,.05);border:1px solid var(--line)}
.stat b{display:block;font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;color:#fff;line-height:1.1}
.stat span{font-size:.74rem;color:var(--mut)}

/* ---------- kartu spesifikasi (container ber-key) ---------- */
.st-key-spec{border:1px solid var(--line)!important;border-radius:22px!important;padding:.5rem .6rem .3rem!important;
background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.025))!important}
.ct{font-size:.74rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--mut);margin:.3rem 0 .5rem}

/* ---------- widget Streamlit ---------- */
label[data-testid="stWidgetLabel"] p{font-size:.78rem!important;font-weight:600!important;color:var(--mut)!important}
div[data-baseweb="select"]>div,div[data-baseweb="base-input"]{background:rgba(255,255,255,.055)!important;border-color:var(--line)!important;border-radius:12px!important}
div[data-baseweb="select"]:focus-within>div,div[data-baseweb="base-input"]:focus-within{border-color:var(--vi)!important;box-shadow:0 0 0 3px rgba(139,92,246,.28)!important}
div[data-baseweb="tab-list"]{gap:6px;background:rgba(255,255,255,.04);border:1px solid var(--line);padding:5px;border-radius:14px}
button[data-baseweb="tab"]{border-radius:10px;padding:.5rem 1.1rem;color:var(--mut);font-weight:700}
button[data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(135deg,rgba(139,92,246,.95),rgba(34,211,238,.75));color:#fff}
div[data-baseweb="tab-highlight"],div[data-baseweb="tab-border"]{display:none}
.stButton>button{border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.05);color:var(--txt);font-weight:600;font-size:.8rem;padding:.25rem .9rem}
.stButton>button:hover{border-color:var(--cy);color:#fff;box-shadow:0 0 0 3px rgba(34,211,238,.18)}
div[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:16px!important;background:rgba(255,255,255,.03)}

/* ---------- kartu hasil ---------- */
.res{border-radius:24px;padding:1.5rem 1.6rem;border:1px solid rgba(139,92,246,.38);margin-top:.9rem;
background:linear-gradient(160deg,rgba(139,92,246,.20),rgba(34,211,238,.08) 60%,rgba(255,255,255,.03));
box-shadow:0 24px 60px -24px rgba(139,92,246,.55);animation:rise .45s ease both}
.rl{font-size:.74rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--mut)}
.rp{font-family:'Space Grotesk',sans-serif;font-weight:700;letter-spacing:-.03em;line-height:1.05;margin:.35rem 0 .25rem;display:flex;align-items:baseline;gap:.35rem;flex-wrap:wrap}
.rp .cur{font-size:1.25rem;color:var(--mut);font-weight:600}
.rp .num{font-size:clamp(2.6rem,6.2vw,3.9rem);background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent}
.rp .unit{font-size:1.25rem;color:var(--mut);font-weight:600}
.rr{font-size:.92rem;color:var(--txt)}
.rr b{font-weight:700}
.rr b,.gl b,.sum b{white-space:nowrap}
.gauge{position:relative;height:10px;border-radius:999px;background:rgba(255,255,255,.09);margin:1.5rem 0 .55rem}
.band{position:absolute;top:0;bottom:0;border-radius:999px;background:var(--grad);opacity:.92;transform-origin:left;animation:grow .8s ease both}
.mk{position:absolute;top:50%;width:18px;height:18px;border-radius:50%;background:#fff;border:3px solid var(--vi);transform:translate(-50%,-50%);box-shadow:0 0 0 6px rgba(139,92,246,.25)}
.mk.pin{border-color:var(--warn);box-shadow:0 0 0 6px rgba(251,191,36,.25)}
.gl{display:flex;justify-content:space-between;gap:10px;font-size:.78rem;color:var(--mut)}
.gl b{color:var(--txt);font-weight:700}
.conf{display:flex;align-items:center;gap:12px;margin:1rem 0 .15rem}
.meter{display:flex;gap:4px}
.meter i{width:28px;height:6px;border-radius:4px;background:rgba(255,255,255,.15)}
.meter i.on{background:var(--c)}
.cl{font-size:.86rem;font-weight:700;color:var(--c)}
.cd{font-size:.8rem;color:var(--mut);line-height:1.55;margin:.2rem 0 0}
.pp{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:1.1rem}
.pc{padding:11px 12px;border-radius:14px;background:rgba(255,255,255,.055);border:1px solid var(--line)}
.pc em{display:block;font-style:normal;font-size:.66rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--mut);margin-bottom:3px}
.pc b{font-family:'Space Grotesk',sans-serif;font-size:1.02rem;color:#fff;font-weight:700}
.pc span{display:block;font-size:.7rem;color:var(--dim);margin-top:2px;line-height:1.4}
.pc.mid{border-color:rgba(139,92,246,.6);background:rgba(139,92,246,.16)}
.cht{margin-top:1rem;padding:15px 15px 11px;border-radius:20px;background:rgba(255,255,255,.04);border:1px solid var(--line)}
.cht h5{margin:0 0 .15rem!important;padding:0!important;font-size:.92rem!important;font-weight:700!important;color:#fff}
.cht p{margin:0 0 .7rem;font-size:.75rem;color:var(--dim);line-height:1.5}
.bars{display:flex;align-items:flex-end;gap:8px;height:190px}
.bc{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:4px;height:100%}
.bv{font-size:.7rem;color:var(--mut);font-family:'Space Grotesk',sans-serif}
.bb{width:100%;border-radius:9px 9px 3px 3px;background:rgba(255,255,255,.14);transform-origin:bottom;animation:grow-y .6s ease both}
.bc.sel .bb{background:var(--grad)}
.bc.sel .bv{color:#fff;font-weight:700}
.by{font-size:.7rem;color:var(--dim)}
.bc.sel .by{color:var(--txt);font-weight:700}

/* ---------- vonis pembeli ---------- */
.vd{display:flex;gap:14px;align-items:flex-start;padding:1.15rem 1.25rem;border-radius:22px;border:1px solid var(--vc);margin-top:.9rem;animation:rise .45s ease both}
.vd.ok{--vc:rgba(52,211,153,.55);background:linear-gradient(160deg,rgba(52,211,153,.17),rgba(255,255,255,.02))}
.vd.low{--vc:rgba(34,211,238,.55);background:linear-gradient(160deg,rgba(34,211,238,.17),rgba(255,255,255,.02))}
.vd.high{--vc:rgba(251,113,133,.55);background:linear-gradient(160deg,rgba(251,113,133,.17),rgba(255,255,255,.02))}
.vi{flex:0 0 auto;width:42px;height:42px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;font-weight:800;color:#07101f}
.vd.ok .vi{background:var(--ok)}.vd.low .vi{background:var(--cy)}.vd.high .vi{background:var(--bad)}
.vt{font-size:1.15rem;font-weight:800;color:#fff;letter-spacing:-.01em}
.vx{font-size:.86rem;color:var(--mut);line-height:1.6;margin-top:.2rem}
.sum{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:.9rem;padding:12px 14px;border-radius:16px;background:rgba(255,255,255,.045);border:1px solid var(--line);font-size:.84rem;color:var(--mut)}
.sum b{color:#fff;font-family:'Space Grotesk',sans-serif}

/* ---------- lain-lain ---------- */
.al{padding:10px 14px;border-radius:12px;font-size:.84rem;line-height:1.55;margin:0 0 10px;border:1px solid}
.al.warn{background:rgba(251,191,36,.10);border-color:rgba(251,191,36,.38);color:#FDE68A}
.al.info{background:rgba(34,211,238,.09);border-color:rgba(34,211,238,.35);color:#A5F3FC}
.stop{padding:1.6rem;border-radius:24px;border:1px solid rgba(251,191,36,.4);background:linear-gradient(160deg,rgba(251,191,36,.12),rgba(255,255,255,.02));margin-top:.9rem}
.stop h3{margin:0 0 .4rem!important;padding:0!important;font-size:1.2rem!important;color:#fff;font-weight:800}
.stop p{color:var(--mut);font-size:.9rem;line-height:1.65;margin:0 0 .9rem}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{padding:6px 13px;border-radius:999px;background:rgba(255,255,255,.07);border:1px solid var(--line);font-size:.8rem;color:var(--txt);font-weight:600}
.tb{width:100%;border-collapse:collapse;font-size:.84rem;margin:.4rem 0 .6rem}
.tb th{text-align:right;color:var(--mut);font-weight:600;padding:6px 8px;border-bottom:1px solid var(--line)}
.tb td{padding:9px 8px;text-align:right;border-bottom:1px solid rgba(255,255,255,.05);color:var(--txt)}
.tb td:first-child,.tb th:first-child{text-align:left}
.ft{margin-top:2rem;padding-top:1.2rem;border-top:1px solid var(--line);color:var(--dim);font-size:.8rem;line-height:1.8;text-align:center}
.ft a{color:var(--cy);text-decoration:none}
@keyframes pulse{0%,100%{box-shadow:0 0 0 4px rgba(52,211,153,.2)}50%{box-shadow:0 0 0 8px rgba(52,211,153,.05)}}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes grow-y{from{transform:scaleY(0)}to{transform:scaleY(1)}}
@media (max-width:640px){
.hero{padding:1.5rem 1.2rem 1.3rem;border-radius:22px}
.res{padding:1.2rem 1.1rem}.pp{gap:6px}.pc{padding:9px 8px}.pc b{font-size:.88rem}.bars{gap:5px}}
</style>
"""


def h(s):
    """Jadikan satu baris tanpa indentasi: Markdown tidak akan salah menafsirkan HTML-nya."""
    return " ".join(line.strip() for line in s.strip().splitlines() if line.strip())


def tampil(html_str):
    st.markdown(html_str if html_str.lstrip().startswith("<style") else h(html_str), unsafe_allow_html=True)


def angka_id(n):
    return f"{n:,.0f}".replace(",", ".")


def rp_bagian(x):
    """(angka, satuan) -> ('135', 'juta') atau ('1,25', 'miliar')."""
    if x >= 1e9:
        return f"{x / 1e9:.2f}".replace(".", ","), "miliar"
    return angka_id(x / 1e6), "juta"


def rp(x):
    a, s = rp_bagian(x)
    return f"Rp {a} {s}"


@st.cache_resource
def muat_model():
    return joblib.load(PATH_MODEL)


def buat_baris(info, merek, model, tahun, km, mesin_l, bahan_bakar, bodi, transmisi):
    lengkap = f"{merek} {model}".strip()
    model_g = lengkap if lengkap in info["model_dikenal"] else f"{merek} LAINNYA"
    return {"merek": merek, "model_g": model_g, "bahan_bakar": bahan_bakar, "bodi": bodi,
            "transmisi": transmisi, "sumber": info["sumber_prediksi"],
            "umur": max(info["tahun_acuan"] - tahun, 0), "km": km,
            "mesin_l": 0.0 if bahan_bakar == "Listrik" else mesin_l}


def prediksi(paket, merek, model, tahun, km, mesin_l, bahan_bakar, bodi, transmisi):
    info = paket["info"]
    lengkap = f"{merek} {model}".strip()
    n = info["jumlah_per_model"].get(lengkap, 0)
    keyakinan = "Tinggi" if n >= 30 else "Sedang" if n >= info["min_model"] else "Rendah"
    n_merek = info["jumlah_per_merek"][merek]
    merek_kecil = n_merek < info["merek_kecil"]
    if merek_kecil:
        keyakinan = "Rendah"
    baris = pd.DataFrame([buat_baris(info, merek, model, tahun, km, mesin_l, bahan_bakar, bodi, transmisi)])
    est = float(paket["est"].predict(baris)[0])
    f_low, f_high = info["faktor_rentang"][keyakinan]
    return {"estimasi": est, "rendah": est * f_low, "tinggi": est * f_high, "keyakinan": keyakinan,
            "n": n, "merek_kecil": merek_kecil, "n_merek": n_merek}


def pola_harga(paket, merek, model, tahun, km, mesin_l, bahan_bakar, bodi, transmisi, sisi=3):
    """Estimasi harga bila tahun mobil berbeda, dengan pemakaian km per tahun yang sama."""
    info = paket["info"]
    acuan = info["tahun_acuan"]
    laju = km / max(acuan - tahun, 1) if not np.isnan(km) else np.nan
    tahun_list = [t for t in range(tahun - sisi, tahun + sisi + 1) if 1990 <= t <= acuan]
    baris = []
    for t in tahun_list:
        km_t = laju * max(acuan - t, 0) if not np.isnan(laju) else np.nan
        baris.append(buat_baris(info, merek, model, t, km_t, mesin_l, bahan_bakar, bodi, transmisi))
    pred = paket["est"].predict(pd.DataFrame(baris))
    return list(zip(tahun_list, [float(p) for p in pred]))


def cek_input(tahun, km, mesin_l, bahan_bakar, tahun_acuan):
    pesan = []
    umur = max(tahun_acuan - tahun, 0)
    if not np.isnan(km):
        if umur >= 2 and km < 3000:
            pesan.append("Kilometer sangat kecil untuk mobil seumur ini. Pastikan angkanya benar.")
        if km / max(umur, 1) > 40000:
            pesan.append("Kilometer per tahun sangat tinggi (di atas 40.000 km/tahun). Pastikan angkanya benar.")
    if bahan_bakar != "Listrik" and mesin_l <= 0:
        pesan.append("Kapasitas mesin 0 liter hanya masuk akal untuk mobil listrik.")
    return pesan


def daftar_model(info, merek):
    """Model yang benar-benar dikenal model (>= min_model iklan), terpopuler dulu, tanpa duplikat huruf besar/kecil."""
    awal = merek + " "
    terbaik = {}
    for k in info["model_dikenal"]:
        if not k.startswith(awal):
            continue
        nama, jml = k[len(awal):], info["jumlah_per_model"][k]
        if nama.lower() == "lainnya":
            continue
        kunci = nama.lower()
        if kunci not in terbaik or jml > terbaik[kunci][1]:
            terbaik[kunci] = (nama, jml)
    return sorted(terbaik.values(), key=lambda t: -t[1])


# ----------------------------------------------------------------------------- potongan HTML
def html_hero(info):
    total = sum(info["jumlah_per_merek"].values())
    mape = info["skor_cv"]["Model dikenal, km diketahui"]["MAPE (%)"]
    return f"""
    <div class="hero">
      <div class="badge"><i></i> Machine learning · Gradient Boosting</div>
      <h1>Berapa harga <span class="grad">mobil bekasmu</span> yang pantas?</h1>
      <p class="sub">Isi spesifikasi mobil, lalu dapatkan estimasi harga, rentang wajar, dan tingkat keyakinan
      model. Dilatih dari ribuan iklan mobil bekas nyata di marketplace Indonesia.</p>
      <div class="stats">
        <div class="stat"><b>{angka_id(total)}</b><span>iklan untuk melatih</span></div>
        <div class="stat"><b>{len(info['merek_didukung'])} merek</b><span>{len(info['model_dikenal'])} model dikenal</span></div>
        <div class="stat"><b>±{mape:.0f}%</b><span>rata-rata meleset (model populer)</span></div>
      </div>
    </div>
    """


WARNA_KEYAKINAN = {"Tinggi": "#34D399", "Sedang": "#FBBF24", "Rendah": "#FB923C"}
SEGMEN = {"Tinggi": 3, "Sedang": 2, "Rendah": 1}
PENJELASAN = {
    "Tinggi": "Model ini punya banyak data ({n} iklan), jadi estimasinya paling bisa dipercaya.",
    "Sedang": "Data model ini lumayan ({n} iklan), tapi belum banyak. Rentang harga bisa lebih lebar.",
    "Rendah": "Model ini jarang atau tidak ada di data, jadi ini estimasi kasar berdasarkan mobil sejenis. "
              "Anggap sebagai gambaran awal, bukan patokan harga.",
}


def html_keyakinan(hasil):
    k = hasil["keyakinan"]
    segmen = "".join(f'<i class="{"on" if i < SEGMEN[k] else ""}"></i>' for i in range(3))
    if hasil["merek_kecil"]:
        teks = (f"Data merek ini masih sedikit ({hasil['n_merek']} iklan), jadi anggap estimasi ini kasar "
                "dan lebar rentangnya.")
    else:
        teks = PENJELASAN[k].format(n=hasil["n"])
    return f"""
    <div class="conf" style="--c:{WARNA_KEYAKINAN[k]}"><div class="meter">{segmen}</div><span class="cl">Keyakinan {k}</span></div>
    <p class="cd">{teks}</p>
    """


def html_gauge(rendah, estimasi, tinggi, harga=None):
    batas = [rendah, tinggi] + ([harga] if harga else [])
    lo, hi = min(batas) * 0.88, max(batas) * 1.12

    def pos(v):
        return max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))

    marker = f'<div class="mk" style="left:{pos(estimasi):.1f}%"></div>'
    if harga:
        marker += f'<div class="mk pin" style="left:{pos(harga):.1f}%"></div>'
    return f"""
    <div class="gauge"><div class="band" style="left:{pos(rendah):.1f}%;width:{pos(tinggi) - pos(rendah):.1f}%"></div>{marker}</div>
    <div class="gl"><span>Batas bawah <b>{rp(rendah)}</b></span><span>Batas atas <b>{rp(tinggi)}</b></span></div>
    """


def html_hasil_penjual(hasil):
    angka, satuan = rp_bagian(hasil["estimasi"])
    return f"""
    <div class="res">
      <div class="rl">Estimasi harga pasang</div>
      <div class="rp"><span class="cur">Rp</span><span class="num">{angka}</span><span class="unit">{satuan}</span></div>
      <div class="rr">Rentang wajar <b>{rp(hasil['rendah'])}</b> sampai <b>{rp(hasil['tinggi'])}</b></div>
      {html_gauge(hasil['rendah'], hasil['estimasi'], hasil['tinggi'])}
      {html_keyakinan(hasil)}
    </div>
    <div class="pp">
      <div class="pc"><em>Cepat laku</em><b>{rp(hasil['rendah'])}</b><span>Dekat batas bawah rentang</span></div>
      <div class="pc mid"><em>Seimbang</em><b>{rp(hasil['estimasi'])}</b><span>Estimasi model</span></div>
      <div class="pc"><em>Ruang tawar</em><b>{rp(hasil['tinggi'])}</b><span>Dekat batas atas rentang</span></div>
    </div>
    """


def html_pola(pola, tahun):
    nilai = [v / 1e6 for _, v in pola]
    maks = max(nilai)
    kolom = ""
    for (t, _), v in zip(pola, nilai):
        tinggi = 16 + 118 * (v / maks)
        kolom += (f'<div class="bc{" sel" if t == tahun else ""}"><span class="bv">{v:,.0f}</span>'
                  f'<div class="bb" style="height:{tinggi:.0f}px"></div><span class="by">{t}</span></div>')
    return f"""
    <div class="cht">
      <h5>Harga menurut tahun mobil</h5>
      <p>Estimasi model (juta Rp) bila tahunnya berbeda, spesifikasi lain sama dan pemakaian per tahun diasumsikan sama.
      Gambaran pola, bukan jaminan.</p>
      <div class="bars">{kolom}</div>
    </div>
    """


def html_vonis(hasil, harga):
    est = hasil["estimasi"]
    selisih = (harga - est) / est * 100
    if harga < hasil["rendah"]:
        tipe, ikon, judul = "low", "↓", "Di bawah rentang wajar"
        teks = (f"{selisih:+.0f}% dari estimasi. Bisa jadi murah, atau ada kondisi yang perlu dicek "
                "(surat, bekas tabrak atau banjir, kilometer diturunkan).")
    elif harga > hasil["tinggi"]:
        tipe, ikon, judul = "high", "↑", "Di atas rentang wajar"
        teks = (f"{selisih:+.0f}% dari estimasi. Layak ditawar, atau cek apakah ada kelebihan khusus "
                "yang tidak ikut dihitung model.")
    else:
        tipe, ikon, judul = "ok", "✓", "Harga wajar"
        teks = f"Masih dalam rentang wajar ({'0' if abs(selisih) < 0.5 else format(selisih, '+.0f')}% dari estimasi)."
    tawar = ""
    if harga > est * 1.02:
        tawar = f" Titik tawar yang masuk akal: sekitar {rp(est)} (selisih {rp(harga - est)})."
    return f"""
    <div class="vd {tipe}"><div class="vi">{ikon}</div><div><div class="vt">{judul}</div><div class="vx">{teks}{tawar}</div></div></div>
    <div class="res" style="margin-top:.9rem;padding:1.1rem 1.3rem">
      <div class="rl">Harga pasang (kuning) dibanding estimasi (ungu)</div>
      {html_gauge(hasil['rendah'], est, hasil['tinggi'], harga)}
    </div>
    <div class="sum"><span>Estimasi model <b>{rp(est)}</b></span><span>Rentang <b>{rp(hasil['rendah'])} - {rp(hasil['tinggi'])}</b></span><span>Keyakinan <b>{hasil['keyakinan']}</b></span></div>
    """


def html_tolak(daftar_merek):
    chips = "".join(f'<span class="chip">{m}</span>' for m in daftar_merek)
    return f"""
    <div class="stop">
      <h3>Merek ini belum didukung</h3>
      <p>Datanya terlalu sedikit untuk menebak harga dengan jujur, jadi aplikasi ini tidak memberi angka. Lebih baik
      tidak ada estimasi daripada estimasi yang menyesatkan. Merek yang didukung saat ini:</p>
      <div class="chips">{chips}</div>
    </div>
    """


def html_tabel_akurasi(skor):
    nama = {"Model dikenal, km diketahui": "Model populer, km diisi",
            "Model dikenal, semua iklan": "Model populer, semua iklan",
            "Model baru, km diketahui": "Model belum pernah dilihat, km diisi",
            "Model baru, semua iklan": "Model belum pernah dilihat, semua iklan"}
    baris = ""
    for kunci, label in nama.items():
        s = skor[kunci]
        koma = lambda x: f"{x:.1f}".replace(".", ",")
        baris += (f"<tr><td>{label}</td><td>{koma(s['MAE (juta Rp)'])}</td><td>{koma(s['MAPE (%)'])}%</td>"
                  f"<td>{koma(s['Median error (%)'])}%</td></tr>")
    return (f'<table class="tb"><tr><th>Skenario uji</th><th>Meleset (juta Rp)</th><th>Meleset (%)</th>'
            f'<th>Median (%)</th></tr>{baris}</table>')


# ----------------------------------------------------------------------------- muat model
try:
    paket = muat_model()
except Exception as e:
    st.error("Model belum bisa dimuat. Jalankan dulu notebook 02_zenodo_model.ipynb sampai selesai, "
             "dan pastikan versi library sama dengan saat model dibuat.")
    st.caption(f"Detail teknis: {e}")
    st.stop()

info = paket["info"]
pilihan = info["pilihan"]
daftar_merek = list(info["merek_didukung"])
bodi_opsi = [b for b in pilihan["bodi"] if b != "Tidak diketahui"]

tampil(CSS)
tampil(html_hero(info))

# ----------------------------------------------------------------------------- state & contoh cepat
CONTOH = {
    "Avanza 2018": dict(merek="Toyota", model="Avanza", tahun=2018, transmisi="Manual", bahan_bakar="Bensin",
                        bodi="MPV", km=60000, km_na=False, mesin=1.3),
    "Brio 2020": dict(merek="Honda", model="Brio", tahun=2020, transmisi="Automatic", bahan_bakar="Bensin",
                      bodi="Hatchback", km=40000, km_na=False, mesin=1.2),
    "Xpander 2021": dict(merek="Mitsubishi", model="Xpander", tahun=2021, transmisi="Automatic",
                         bahan_bakar="Bensin", bodi="MPV", km=30000, km_na=False, mesin=1.5),
    "BYD Atto 3": dict(merek="BYD", model="Atto 3", tahun=2024, transmisi="Automatic", bahan_bakar="Listrik",
                       bodi="SUV", km=20000, km_na=False, mesin=0.0),
    "C-Class 2018": dict(merek="Mercedes-Benz", model="C-Class", tahun=2018, transmisi="Automatic",
                         bahan_bakar="Bensin", bodi="Sedan", km=50000, km_na=False, mesin=2.0),
}
for k, v in CONTOH["Avanza 2018"].items():
    st.session_state.setdefault(k, v)


def pakai_contoh(nama):
    st.session_state.update(CONTOH[nama])
    st.session_state.pop("harga_pasang", None)


def kartu(nama):
    try:
        return st.container(border=True, key=nama)
    except TypeError:            # Streamlit versi lama belum punya parameter key
        return st.container(border=True)


# ----------------------------------------------------------------------------- tata letak utama
kiri, kanan = st.columns([5, 6], gap="large")

with kiri:
    with kartu("spec"):
        tampil('<div class="ct">Spesifikasi mobil</div>')
        with st.expander("Coba contoh cepat"):
            kol = st.columns(3)
            for i, nama in enumerate(CONTOH):
                kol[i % 3].button(nama, key=f"contoh_{i}", on_click=pakai_contoh, args=(nama,))

        merek = st.selectbox("Merek", daftar_merek + [MEREK_LAIN], key="merek")
        merek_ok = merek != MEREK_LAIN

        if merek_ok:
            opsi_model = daftar_model(info, merek)
            nama_model = [m for m, _ in opsi_model]
            jml_model = dict(opsi_model)
            semua_opsi = nama_model + [MODEL_LAIN]
            if st.session_state.get("model") not in semua_opsi:
                st.session_state["model"] = semua_opsi[0]
            model_pilih = st.selectbox(
                "Model", semua_opsi, key="model",
                format_func=lambda m: m if m == MODEL_LAIN else f"{m}  ·  {angka_id(jml_model[m])} iklan",
                help="Hanya model yang cukup banyak datanya yang tampil. Model lain bisa dipilih lewat opsi terakhir "
                     "(estimasi lebih kasar).")
            model = "" if model_pilih == MODEL_LAIN else model_pilih

            c1, c2 = st.columns(2)
            with c1:
                tahun = st.number_input("Tahun mobil", min_value=1985, max_value=info["tahun_acuan"], step=1, key="tahun")
            with c2:
                transmisi = st.selectbox("Transmisi", pilihan["transmisi"], key="transmisi")

            c3, c4 = st.columns(2)
            with c3:
                bahan_bakar = st.selectbox("Bahan bakar", pilihan["bahan_bakar"], key="bahan_bakar")
            with c4:
                bodi = st.selectbox("Tipe bodi", bodi_opsi, key="bodi")

            listrik = bahan_bakar == "Listrik"
            if listrik:
                st.session_state["mesin"] = 0.0
            elif st.session_state.get("mesin", 0.0) <= 0:
                st.session_state["mesin"] = 1.5

            c5, c6 = st.columns(2)
            with c5:
                km_na = st.checkbox("Kilometer tidak diketahui", key="km_na")
                km_input = st.number_input("Kilometer (km)", min_value=0, max_value=600000, step=1000,
                                           key="km", disabled=km_na)
            with c6:
                mesin_input = st.number_input("Kapasitas mesin (liter)", min_value=0.0, max_value=7.0, step=0.1,
                                              format="%.1f", key="mesin", disabled=listrik,
                                              help="Mobil listrik otomatis 0. Contoh: 1.497 cc diisi 1,5.")
            km = np.nan if km_na else float(km_input)
            mesin_l = 0.0 if listrik else float(mesin_input)

with kanan:
    if not merek_ok:
        tampil(html_tolak(daftar_merek))
    else:
        tahun = int(tahun)
        for pesan in cek_input(tahun, km, mesin_l, bahan_bakar, info["tahun_acuan"]):
            tampil(f'<div class="al warn">{pesan}</div>')
        if km_na:
            tampil('<div class="al info">Tanpa kilometer, akurasi estimasi turun. Isi kilometer kalau kamu tahu.</div>')

        hasil = prediksi(paket, merek, model, tahun, km, mesin_l, bahan_bakar, bodi, transmisi)
        tab_jual, tab_beli = st.tabs(["Saya penjual", "Saya pembeli"])

        with tab_jual:
            tampil(html_hasil_penjual(hasil))
            pola = pola_harga(paket, merek, model, tahun, km, mesin_l, bahan_bakar, bodi, transmisi)
            if len(pola) >= 4:
                tampil(html_pola(pola, tahun))
            st.caption("Rentang dihitung dari seberapa jauh tebakan model meleset pada data uji, dan mencakup sekitar "
                       "4 dari 5 harga iklan mobil serupa.")

        with tab_beli:
            st.session_state.setdefault("harga_pasang", int(round(hasil["estimasi"] / 1e6)))
            harga_pasang = st.number_input("Harga yang dipasang penjual (juta Rp)", min_value=10, max_value=20000,
                                           step=5, key="harga_pasang")
            tampil(html_vonis(hasil, harga_pasang * 1e6))
            st.caption("Ini bantuan pembanding, bukan pengganti cek fisik dan surat mobil.")

# ----------------------------------------------------------------------------- info tambahan
st.write("")
e1, e2 = st.columns(2, gap="large")
with e1:
    with st.expander("Seberapa akurat estimasi ini?"):
        st.markdown("Rata-rata meleset dari harga iklan sebenarnya, diuji dengan validasi silang pada data yang "
                    "tidak dipakai melatih:")
        tampil(html_tabel_akurasi(info["skor_cv"]))
        st.markdown("**Model populer** artinya model mobil itu banyak ada di data latih. **Model belum pernah "
                    "dilihat** artinya seluruh model itu sengaja disembunyikan dari data latih. Makin sedikit data "
                    "sebuah model, makin besar meleset-nya.")
with e2:
    with st.expander("Batasan & sumber data"):
        st.markdown(
            "- Harga yang dipakai adalah **harga iklan**, bukan harga transaksi akhir.\n"
            "- Data utama dari Jabodetabek dan hanya merek tertentu; merek lain belum didukung.\n"
            "- Merek dengan data sedikit (Nissan, Mazda) selalu diberi keyakinan Rendah.\n"
            "- Kondisi fisik, riwayat servis, dan kelengkapan surat tidak ikut dihitung.\n"
            "- Sumber data: Thania, I. & Oktavia, T. (2026), Zenodo, DOI 10.5281/zenodo.21792616, CC BY 4.0; "
            "dan \"Used Car Listings in Indonesia\" (Kaggle, CC0, data dari carsome.id 2023)."
        )

tautan = []
if LINK_GITHUB:
    tautan.append(f'<a href="{LINK_GITHUB}" target="_blank">GitHub</a>')
if LINK_PORTOFOLIO:
    tautan.append(f'<a href="{LINK_PORTOFOLIO}" target="_blank">Portofolio</a>')
pembuat = f"Dibuat oleh {NAMA_PEMBUAT}" if NAMA_PEMBUAT else "Project portofolio data science"
tampil(f'<div class="ft">{pembuat}{" · " + " · ".join(tautan) if tautan else ""}<br>'
       'Estimasi hanya perkiraan statistik dari iklan marketplace, bukan penilaian resmi atau jaminan harga.</div>')
