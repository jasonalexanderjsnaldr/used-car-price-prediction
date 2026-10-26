"""
Estimasi Harga Mobil Bekas (v3)
Jalankan dari folder paling luar project:   streamlit run app/app.py
"""
import base64
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

NAMA_PEMBUAT = "Jason Alexander"
LINK_GITHUB = "https://github.com/jasonalexanderjsnaldr/used-car-price-prediction"
LINK_PORTOFOLIO = "https://jsn-aldr-jasons.vercel.app/"

PATH_MODEL = Path(__file__).resolve().parent.parent / "models" / "model_v2.joblib"
MODEL_LAIN = "Model lain / tidak ada di daftar"
MEREK_LAIN = "Merek lain (BMW, Hyundai, Kia, dll)"

INSIGHT = {
    "iklan": 18884,
    "retensi": [100, 96, 89, 83, 76, 71, 67, 61, 57, 52, 48, 44, 39, 34, 29],
    "merek": [("Daihatsu", 3.6), ("Suzuki", 5.5), ("Mitsubishi", 5.5), ("Honda", 6.1), ("Toyota", 6.8),
              ("Mercedes-Benz", 12.2)],
    "rupiah": [("Daihatsu Sigra", 3.8), ("Honda Brio", 6.5), ("Toyota Avanza", 12.2), ("Kijang Innova", 24.5),
               ("Toyota Fortuner", 29.8), ("Honda CR-V", 42.9), ("Mercedes C-Class", 68.9),
               ("Mercedes E-Class", 83.3)],
}

st.set_page_config(page_title="Estimasi Harga Mobil Bekas", page_icon="🚗", layout="wide",
                   initial_sidebar_state="collapsed")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
@property --num{syntax:'<integer>';inherits:false;initial-value:0}
:root{--line:rgba(255,255,255,.09);--txt:#E9EDF7;--mut:#93A0BD;--dim:#6B7794;--cy:#22D3EE;--vi:#8B5CF6;--pk:#F472B6;
--ok:#34D399;--warn:#FBBF24;--bad:#FB7185;--org:#FB923C;--grad:linear-gradient(90deg,#22D3EE,#8B5CF6 55%,#F472B6)}
.stApp{font-family:'Plus Jakarta Sans',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--txt);
background:radial-gradient(1000px 520px at 8% -8%,rgba(139,92,246,.26),transparent 62%),
radial-gradient(800px 460px at 100% 2%,rgba(34,211,238,.16),transparent 58%),
radial-gradient(700px 400px at 50% 110%,rgba(244,114,182,.10),transparent 60%),#070B16}
.stApp p,.stApp label,.stApp button,.stApp input,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp table{font-family:'Plus Jakarta Sans',system-ui,sans-serif}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer,[data-testid="stDecoration"]{display:none!important}
.block-container{max-width:1180px;padding:1.4rem 1.2rem 3rem}

.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:30px;padding:2.4rem 2.4rem 2rem;margin:0 0 1.3rem;
background:linear-gradient(135deg,rgba(139,92,246,.18),rgba(34,211,238,.07) 55%,rgba(244,114,182,.11))}
.hero:before{content:"";position:absolute;top:-45%;right:-8%;width:560px;height:560px;border-radius:50%;
background:radial-gradient(circle,rgba(34,211,238,.26),transparent 65%);pointer-events:none;animation:drift 9s ease-in-out infinite alternate}
.hero:after{content:"";position:absolute;inset:0;pointer-events:none;
background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);
background-size:34px 34px;-webkit-mask-image:radial-gradient(circle at 78% 20%,#000,transparent 70%);mask-image:radial-gradient(circle at 78% 20%,#000,transparent 70%)}
.hero>*{position:relative;z-index:1}
.hero .art{position:absolute;right:2%;bottom:8%;width:min(46%,470px);z-index:0;opacity:.95;filter:drop-shadow(0 0 22px rgba(139,92,246,.55));animation:float 6s ease-in-out infinite alternate}
.hero .road{position:absolute;left:0;right:0;bottom:0;height:3px;z-index:0;opacity:.7;
background:repeating-linear-gradient(90deg,rgba(255,255,255,.55) 0 34px,transparent 34px 68px);animation:road 1.4s linear infinite}
.badge{display:inline-flex;align-items:center;gap:8px;font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
color:#CFE8F6;background:rgba(255,255,255,.07);border:1px solid var(--line);padding:6px 13px;border-radius:999px}
.badge i{width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 4px rgba(52,211,153,.2);animation:pulse 2s infinite}
.hero h1{font-size:clamp(1.9rem,4.6vw,3.3rem)!important;line-height:1.08!important;font-weight:800!important;letter-spacing:-.03em!important;
margin:.95rem 0 .6rem!important;padding:0!important;color:#fff;max-width:640px}
.grad{background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent}
.hero p.sub{max-width:560px;color:var(--mut);font-size:1.02rem;line-height:1.65;margin:0 0 1.4rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;max-width:620px}
.stat{padding:12px 16px;border-radius:16px;background:rgba(255,255,255,.055);border:1px solid var(--line);backdrop-filter:blur(6px)}
.stat b{display:block;font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;color:#fff;line-height:1.1}
.stat span{font-size:.74rem;color:var(--mut)}

.st-key-spec{border:1px solid var(--line)!important;border-radius:24px!important;padding:.5rem .6rem .3rem!important;
background:linear-gradient(180deg,rgba(255,255,255,.065),rgba(255,255,255,.025))!important}
.ct{font-size:.74rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--mut);margin:.3rem 0 .5rem}
.how{margin-top:1rem;padding:1.1rem 1.2rem .6rem;border-radius:22px;border:1px solid var(--line);background:rgba(255,255,255,.035)}
.how .st{display:flex;gap:12px;margin:0 0 .9rem}
.how .st i{flex:0 0 auto;width:26px;height:26px;border-radius:9px;background:var(--grad);color:#07101f;font-style:normal;font-weight:800;font-size:.8rem;display:flex;align-items:center;justify-content:center}
.how .st b{display:block;color:#fff;font-size:.88rem}
.how .st span{display:block;color:var(--mut);font-size:.76rem;line-height:1.55}

label[data-testid="stWidgetLabel"] p{font-size:.78rem!important;font-weight:600!important;color:var(--mut)!important}
div[data-baseweb="select"]>div,div[data-baseweb="base-input"]{background:rgba(255,255,255,.055)!important;border-color:var(--line)!important;border-radius:12px!important}
div[data-baseweb="select"]:focus-within>div,div[data-baseweb="base-input"]:focus-within{border-color:var(--vi)!important;box-shadow:0 0 0 3px rgba(139,92,246,.28)!important}
div[role="radiogroup"]{gap:6px!important;flex-wrap:wrap}
div[role="radiogroup"]>label{background:rgba(255,255,255,.055);border:1px solid var(--line);border-radius:999px;padding:.3rem .85rem .3rem .6rem;margin:0!important;cursor:pointer}
div[role="radiogroup"]>label:has(input:checked){background:linear-gradient(135deg,rgba(139,92,246,.95),rgba(34,211,238,.75));border-color:transparent;color:#fff}
div[role="radiogroup"]>label:has(input:checked) *{color:#fff!important}
div[data-baseweb="tab-list"]{gap:6px;background:rgba(255,255,255,.04);border:1px solid var(--line);padding:5px;border-radius:14px}
button[data-baseweb="tab"]{border-radius:10px;padding:.5rem 1.05rem;color:var(--mut);font-weight:700}
button[data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(135deg,rgba(139,92,246,.95),rgba(34,211,238,.75));color:#fff}
div[data-baseweb="tab-highlight"],div[data-baseweb="tab-border"]{display:none}
.stButton>button,.stDownloadButton>button{border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.05);color:var(--txt);font-weight:600;font-size:.82rem;padding:.3rem 1rem}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--cy);color:#fff;box-shadow:0 0 0 3px rgba(34,211,238,.18)}
div[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:16px!important;background:rgba(255,255,255,.03)}

.res{position:relative;overflow:hidden;border-radius:26px;padding:1.5rem 1.6rem;border:1px solid rgba(139,92,246,.4);margin-top:.9rem;
background:linear-gradient(160deg,rgba(139,92,246,.22),rgba(34,211,238,.09) 60%,rgba(255,255,255,.03));
box-shadow:0 26px 70px -26px rgba(139,92,246,.6);animation:rise .5s ease both}
.res .carart{position:absolute;right:14px;top:10px;width:min(42%,220px);opacity:.92;pointer-events:none}
.res>*:not(.carart){position:relative}
.rl{font-size:.74rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--mut)}
.rp{font-family:'Space Grotesk',sans-serif;font-weight:700;letter-spacing:-.03em;line-height:1.05;margin:.35rem 0 .25rem;display:flex;align-items:baseline;gap:.35rem;flex-wrap:wrap}
.rp .cur,.rp .unit{font-size:1.25rem;color:var(--mut);font-weight:600}
.rp .num{font-size:clamp(2.7rem,6.4vw,4rem);background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent}
.cnt{counter-reset:n var(--num);animation:cnt 1.2s cubic-bezier(.2,.7,.2,1) forwards}
.cnt:after{content:counter(n)}
.rr{font-size:.92rem;color:var(--txt)}
.rr b,.gl b,.sum b{font-weight:700;white-space:nowrap}
.gauge{position:relative;height:10px;border-radius:999px;background:rgba(255,255,255,.09);margin:1.5rem 0 .55rem}
.band{position:absolute;top:0;bottom:0;border-radius:999px;background:var(--grad);opacity:.92;transform-origin:left;animation:grow .9s ease both}
.mk{position:absolute;top:50%;width:18px;height:18px;border-radius:50%;background:#fff;border:3px solid var(--vi);transform:translate(-50%,-50%);box-shadow:0 0 0 6px rgba(139,92,246,.25)}
.mk.pin{border-color:var(--warn);box-shadow:0 0 0 6px rgba(251,191,36,.25)}
.gl{display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px 10px;font-size:.78rem;color:var(--mut)}
.gl b{color:var(--txt)}
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

.proj{margin-top:1rem;padding:16px 16px 14px;border-radius:22px;background:rgba(255,255,255,.04);border:1px solid var(--line)}
.proj h5{margin:0 0 .15rem!important;padding:0!important;font-size:.95rem!important;font-weight:700!important;color:#fff}
.proj>p{margin:0 0 .8rem;font-size:.75rem;color:var(--dim);line-height:1.5}
.pbars{display:flex;align-items:flex-end;gap:8px;height:170px}
.pcol{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:3px;height:100%}
.pv{font-family:'Space Grotesk',sans-serif;font-size:.74rem;color:var(--mut);font-weight:600}
.pb{width:100%;border-radius:10px 10px 3px 3px;background:linear-gradient(180deg,rgba(139,92,246,.6),rgba(34,211,238,.22));transform-origin:bottom;animation:grow-y .7s ease both}
.pcol.now .pb{background:var(--grad)}
.pcol.now .pv{color:#fff;font-weight:700}
.py{font-size:.7rem;color:var(--dim);white-space:nowrap}
.pcol.now .py{color:var(--txt);font-weight:700}
.pd{font-size:.66rem;color:var(--bad);min-height:.9rem}
.pst{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:.9rem}
.pst .pc b{font-size:.95rem}

.vcard{display:grid;grid-template-columns:300px 1fr;gap:1.2rem;align-items:center;padding:1.2rem 1.3rem;border-radius:26px;border:1px solid var(--vc);margin-top:.9rem;animation:rise .5s ease both}
.vcard.ok{--vc:rgba(52,211,153,.55);background:linear-gradient(160deg,rgba(52,211,153,.17),rgba(255,255,255,.02))}
.vcard.low{--vc:rgba(34,211,238,.55);background:linear-gradient(160deg,rgba(34,211,238,.17),rgba(255,255,255,.02))}
.vcard.high{--vc:rgba(251,113,133,.55);background:linear-gradient(160deg,rgba(251,113,133,.17),rgba(255,255,255,.02))}
.dial{position:relative;width:100%;aspect-ratio:2/1}
.dial .arc{position:absolute;inset:0;border-radius:999px 999px 0 0;
background:conic-gradient(from -90deg at 50% 100%,#22D3EE 0deg 58deg,#34D399 62deg 118deg,#FB7185 122deg 180deg);
-webkit-mask:radial-gradient(circle farthest-side at 50% 100%,transparent 0 63%,#000 63.5%);mask:radial-gradient(circle farthest-side at 50% 100%,transparent 0 63%,#000 63.5%)}
.needle{position:absolute;left:50%;bottom:0;width:5px;height:84%;margin-left:-2.5px;border-radius:5px;background:linear-gradient(#fff,rgba(255,255,255,.15));
transform-origin:50% 100%;transform:rotate(var(--a));animation:sweep 1.3s cubic-bezier(.3,1.35,.4,1) both;filter:drop-shadow(0 0 6px rgba(255,255,255,.6))}
.hub{position:absolute;left:50%;bottom:-9px;width:22px;height:22px;margin-left:-11px;border-radius:50%;background:#fff;box-shadow:0 0 0 5px rgba(255,255,255,.18)}
.dc{text-align:center;margin-top:.55rem}
.dc b{display:block;font-size:1.5rem;font-weight:800;color:#fff;letter-spacing:-.01em;line-height:1.1}
.dc span{font-size:.74rem;color:var(--mut)}
.dlab{display:flex;justify-content:space-between;font-size:.66rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--dim);margin-top:1rem;padding:0 2%}
.vt{font-size:1.2rem;font-weight:800;color:#fff;letter-spacing:-.01em}
.vx{font-size:.86rem;color:var(--mut);line-height:1.6;margin-top:.3rem}
.sum{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:.9rem;padding:12px 14px;border-radius:16px;background:rgba(255,255,255,.045);border:1px solid var(--line);font-size:.84rem;color:var(--mut)}
.sum b{color:#fff;font-family:'Space Grotesk',sans-serif}
.plan{margin-top:.9rem;padding:1.2rem 1.3rem;border-radius:22px;border:1px solid rgba(251,191,36,.35);background:linear-gradient(160deg,rgba(251,191,36,.10),rgba(255,255,255,.02));animation:rise .5s ease both}
.plan p{margin:.7rem 0 0;font-size:.75rem;color:var(--dim);line-height:1.55}

.ins-h{font-size:1.05rem;font-weight:800;color:#fff;margin:.3rem 0 .15rem}
.ins-s{font-size:.8rem;color:var(--mut);margin:0 0 .9rem;line-height:1.55}
.kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:.4rem 0 1rem}
.kpi div{padding:14px;border-radius:18px;background:rgba(255,255,255,.055);border:1px solid var(--line)}
.kpi b{display:block;font-family:'Space Grotesk',sans-serif;font-size:1.55rem;color:#fff;line-height:1.1}
.kpi span{font-size:.72rem;color:var(--mut);line-height:1.4;display:block;margin-top:3px}
.card2{margin-top:.9rem;padding:16px;border-radius:22px;background:rgba(255,255,255,.04);border:1px solid var(--line)}
.card2 h5{margin:0 0 .15rem!important;padding:0!important;font-size:.95rem!important;font-weight:700!important;color:#fff}
.card2>p{margin:0 0 .8rem;font-size:.75rem;color:var(--dim);line-height:1.5}
.rbars{display:flex;align-items:flex-end;gap:5px;height:150px}
.rb{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:3px;height:100%}
.rb i{width:100%;border-radius:7px 7px 2px 2px;background:linear-gradient(180deg,#8B5CF6,rgba(34,211,238,.55));transform-origin:bottom;animation:grow-y .7s ease both}
.rb span{font-size:.62rem;color:var(--dim)}
.rb em{font-style:normal;font-size:.62rem;color:var(--mut);font-weight:600}
.hb{display:grid;grid-template-columns:118px 1fr 52px;gap:10px;align-items:center;margin:7px 0;font-size:.8rem}
.hb .hl{color:var(--txt)}
.hb .ht{height:9px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden}
.hb .hf{height:100%;border-radius:99px;background:var(--grad);transform-origin:left;animation:grow .8s ease both}
.hb .hv{text-align:right;font-family:'Space Grotesk',sans-serif;color:#fff;font-weight:600;font-size:.82rem;white-space:nowrap}
.hb.hot .hf{background:linear-gradient(90deg,#F472B6,#FB7185)}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{padding:6px 13px;border-radius:999px;background:rgba(255,255,255,.07);border:1px solid var(--line);font-size:.8rem;color:var(--txt);font-weight:600}
.chip b{color:var(--cy)}
.note{margin-top:1rem;font-size:.74rem;color:var(--dim);line-height:1.6}
.note a{color:var(--cy);text-decoration:none;font-weight:600}

.al{padding:10px 14px;border-radius:12px;font-size:.84rem;line-height:1.55;margin:0 0 10px;border:1px solid}
.al.warn{background:rgba(251,191,36,.10);border-color:rgba(251,191,36,.38);color:#FDE68A}
.al.info{background:rgba(34,211,238,.09);border-color:rgba(34,211,238,.35);color:#A5F3FC}
.stop{padding:1.6rem;border-radius:24px;border:1px solid rgba(251,191,36,.4);background:linear-gradient(160deg,rgba(251,191,36,.12),rgba(255,255,255,.02));margin-top:.9rem}
.stop h3{margin:0 0 .4rem!important;padding:0!important;font-size:1.2rem!important;color:#fff;font-weight:800}
.stop p{color:var(--mut);font-size:.9rem;line-height:1.65;margin:0 0 .9rem}
.tb{width:100%;border-collapse:collapse;font-size:.84rem;margin:.4rem 0 .6rem}
.tb th{text-align:right;color:var(--mut);font-weight:600;padding:6px 8px;border-bottom:1px solid var(--line)}
.tb td{padding:9px 8px;text-align:right;border-bottom:1px solid rgba(255,255,255,.05);color:var(--txt)}
.tb td:first-child,.tb th:first-child{text-align:left}
.ft{margin-top:2rem;padding-top:1.2rem;border-top:1px solid var(--line);color:var(--dim);font-size:.8rem;line-height:1.8;text-align:center}
.ft a{color:var(--cy);text-decoration:none}
@keyframes cnt{from{--num:0}to{--num:var(--to)}}
@keyframes sweep{from{transform:rotate(-90deg)}to{transform:rotate(var(--a))}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 4px rgba(52,211,153,.2)}50%{box-shadow:0 0 0 8px rgba(52,211,153,.05)}}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes grow-y{from{transform:scaleY(0)}to{transform:scaleY(1)}}
@keyframes float{from{transform:translateY(0)}to{transform:translateY(-8px)}}
@keyframes drift{from{transform:translate(0,0)}to{transform:translate(-40px,30px)}}
@keyframes road{from{background-position:0 0}to{background-position:-68px 0}}
@media (max-width:760px){
.hero{padding:1.5rem 1.2rem 1.4rem;border-radius:24px}.hero .art{display:none}.how{display:none}
.res{padding:1.2rem 1.1rem}.res .carart{width:38%;opacity:.55}
.pp,.pst{gap:6px}.pc{padding:9px 8px}.pc b{font-size:.88rem}.pbars,.rbars{gap:4px}
.vcard{grid-template-columns:1fr;padding:1.1rem}.kpi b{font-size:1.25rem}.hb{grid-template-columns:96px 1fr 48px}}
.tick{overflow:hidden;margin-top:1.3rem;border-block:1px solid var(--line);padding:9px 0;-webkit-mask-image:linear-gradient(90deg,transparent,#000 8%,#000 92%,transparent);mask-image:linear-gradient(90deg,transparent,#000 8%,#000 92%,transparent)}
.tick>div{display:flex;gap:2.4rem;width:max-content;animation:marq 30s linear infinite}
.tick:hover>div{animation-play-state:paused}
.tick span{font-size:.76rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);white-space:nowrap}
.tick span:before{content:"\\25C6";margin-right:2.4rem;color:var(--cy)}
.stat,.pc,.card2,.kpi div,.chip{transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease}
.stat:hover,.pc:hover,.card2:hover,.kpi div:hover{transform:translateY(-4px)}
.res:after{content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(110deg,transparent 40%,rgba(255,255,255,.09) 50%,transparent 60%);transform:translateX(-100%);animation:sheen 5s ease-in-out infinite}
@keyframes marq{to{transform:translateX(-50%)}}
@keyframes sheen{0%,60%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
</style>
"""

MANTAP_CSS = '<style>\n:root{--ink:#0D1530;--txt:#1F2A4D;--mut:#55608A;--dim:#7B86A5;--line:rgba(13,21,48,.14);--acc:#2F6FD0;--cy:#2F6FD0;--vi:#2F6FD0;--pk:#F59E0B;--grad:linear-gradient(90deg,#2F6FD0,#F59E0B);--surf:#fff;--card:#fff;--track:#E6E9F2;--hi:#FFF0B8;--onacc:#fff}\n.stApp{background:radial-gradient(800px 480px at 12% 8%,#FFE98F,transparent 62%),radial-gradient(700px 400px at 95% 60%,#FFC83A,transparent 60%),#FACF35!important}\n.block-container{max-width:1240px}\n.mh h1{font-size:clamp(2.5rem,5.6vw,4.7rem)!important;line-height:1.02!important;font-weight:800!important;letter-spacing:-.04em!important;color:#0D1530!important;margin:.3rem 0 .5rem!important;padding:0!important}\n.mh h1 em{font-style:normal;background:#0D1530;color:#FACF35;padding:0 .16em;border-radius:.2em;display:inline-block;transform:rotate(-1.8deg);animation:wob 3s ease-in-out infinite alternate}\n.mh p.sub{font-size:1.3rem;color:#0D1530;font-weight:500;margin:0 0 .6rem}\n.stage{position:relative;height:290px;margin:.6rem 0 1.4rem}\n.stage img.car{position:absolute;left:3%;bottom:16px;width:80%;animation:drive 3s ease-in-out infinite alternate;filter:drop-shadow(0 24px 16px rgba(90,55,0,.4))}\n.stage .road{position:absolute;left:0;right:0;bottom:8px;height:4px;border-radius:4px;background:repeating-linear-gradient(90deg,#0D1530 0 34px,transparent 34px 68px);animation:road .9s linear infinite;opacity:.75}\n.coin{position:absolute;width:76px;height:76px;border-radius:50%;background:radial-gradient(circle at 32% 28%,#FFF6B8,#F7B500 55%,#B97800);border:4px solid #E09A00;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.2rem;color:#8A5500;box-shadow:0 14px 24px rgba(120,70,0,.35);animation:bob 3s ease-in-out infinite}\n.coin.c1{right:4%;top:2%}.coin.c2{left:0;top:26%;width:54px;height:54px;font-size:.9rem;animation-delay:-1s}.coin.c3{right:16%;bottom:0;width:96px;height:96px;font-size:1.5rem;animation-delay:-2s}\n.perks{display:flex;flex-wrap:wrap;gap:14px 26px}\n.perks div{display:flex;align-items:center;gap:10px;font-weight:800;color:#0D1530;font-size:.92rem;line-height:1.2}\n.perks i{font-style:normal;width:46px;height:46px;border-radius:15px;background:#0D1530;display:flex;align-items:center;justify-content:center;font-size:1.3rem;transition:transform .25s}\n.perks div:hover i{transform:rotate(-8deg) scale(1.12)}\n.st-key-spec{position:relative;border:0!important;border-radius:24px!important;box-shadow:0 34px 60px -22px rgba(80,50,0,.55);padding:1.3rem 1.4rem 1rem 3.3rem!important}\n.st-key-spec:before{content:"";position:absolute;left:27px;top:56px;bottom:50px;width:4px;border-radius:4px;background:repeating-linear-gradient(180deg,#2F6FD0 0 10px,transparent 10px 20px);animation:flow 1s linear infinite}\n.st-key-spec:after{content:"\\2713";position:absolute;left:14px;bottom:18px;width:30px;height:30px;border-radius:50%;border:3px solid #2F6FD0;color:#2F6FD0;background:#fff;display:flex;align-items:center;justify-content:center;font-weight:800}\n.st-key-spec .ct{position:relative;font-size:1.05rem;color:#0D1530;text-transform:none;letter-spacing:0;font-weight:800}\n.st-key-spec .ct:before{content:"\\2605";position:absolute;left:-2.6rem;top:-5px;width:30px;height:30px;border-radius:50%;background:#2F6FD0;color:#fff;display:flex;align-items:center;justify-content:center;font-size:.8rem}\n.res,.vcard,.plan,.proj,.card2{border:0!important;box-shadow:0 26px 50px -24px rgba(80,50,0,.5)}\n.res .carart{filter:brightness(0) opacity(.12)}\n.how{display:none}\n.ft,.ft a{color:#0D1530!important}\n@keyframes bob{50%{transform:translateY(-16px) rotate(9deg)}}\n@keyframes drive{from{transform:translateX(-12px) rotate(-.5deg)}to{transform:translateX(16px) rotate(.5deg)}}\n@keyframes wob{to{transform:rotate(1.2deg) scale(1.03)}}\n@keyframes flow{to{background-position:0 20px}}\n@media (max-width:760px){.stage{height:210px}.coin.c3{display:none}}\n.stat,.pc,.card2,.proj,.how,.kpi div,.sum,.chip,.badge,.st-key-spec{background:var(--surf)!important}\n.res,.vcard.ok,.vcard.low,.vcard.high,.plan,.stop,.hero{background:var(--card)!important}\n.pc.mid{background:var(--hi)!important}\n.gauge,.hb .ht,.meter i{background:var(--track)}\n.hero h1,.stat b,.how .st b,.pc b,.pv,.dc b,.vt,.sum b,.ins-h,.kpi b,.card2 h5,.proj h5,.hb .hv,.stop h3,.proj>p b,.badge{color:var(--ink)!important}\n.hero:before,.hero:after,.res:after{display:none}\n.hero .art,.res .carart{filter:brightness(0)}\n.hero .road{background:repeating-linear-gradient(90deg,var(--ink) 0 34px,transparent 34px 68px)}\n.needle{background:linear-gradient(var(--ink),rgba(0,0,0,.2));filter:none}\n.hub{background:var(--ink);box-shadow:0 0 0 5px var(--line)}\n.chip b,.note a,.ft a{color:var(--acc)}\n.tb td{color:var(--txt);border-bottom:1px solid var(--line)}\n.al.warn{background:#FFF1C9;border-color:#C98A00;color:#6B4400}\n.al.info{background:#E3F0F7;border-color:#2B7A9B;color:#0B4A5E}\n.stApp input,.stApp div[data-baseweb="select"] *,.stApp summary,.stApp summary *{color:var(--ink)!important}\ndiv[data-baseweb="select"]>div,div[data-baseweb="base-input"]{background:#fff!important}\ndiv[data-baseweb="popover"] ul,div[data-baseweb="popover"] li{background:#fff!important;color:#111!important}\ndiv[data-baseweb="popover"] li *{color:#111!important}\ndiv[data-baseweb="tab-list"]{background:var(--surf)}\nbutton[data-baseweb="tab"][aria-selected="true"],div[role="radiogroup"]>label:has(input:checked){background:var(--ink)!important;color:var(--onacc)}\ndiv[role="radiogroup"]>label{background:#fff}\n.stButton>button,.stDownloadButton>button{background:#fff;color:var(--ink)}\n</style>'
MANTAP_FIX = r'''
.stApp [data-baseweb="select"] *,.stApp [data-baseweb="input"] *,.stApp [data-baseweb="base-input"] *,.stApp [data-testid="stNumberInputContainer"],.stApp [data-testid="stNumberInputContainer"] *{background-color:#fff!important;color:#0D1530!important;-webkit-text-fill-color:#0D1530!important}
.stApp [data-baseweb="select"]>div,.stApp [data-baseweb="input"]{border:1.5px solid #C9D1E8!important;border-radius:12px!important}
.stApp [data-baseweb="select"]:focus-within>div,.stApp [data-baseweb="input"]:focus-within{border-color:#2F6FD0!important;box-shadow:0 0 0 3px rgba(47,111,208,.22)!important}
.stApp input:disabled{opacity:.45}
div[data-baseweb="popover"] *{background-color:#fff!important;color:#0D1530!important}
div[data-baseweb="popover"] li:hover,div[data-baseweb="popover"] li:hover *,div[data-baseweb="popover"] [aria-selected="true"],div[data-baseweb="popover"] [aria-selected="true"] *{background-color:#FFF0B8!important}
.stApp [data-testid="stWidgetLabel"] p{color:#0D1530!important;font-weight:700!important}
label[data-baseweb="radio"]{background:#fff!important;border:1.5px solid #C9D1E8!important;border-radius:999px!important;padding:.35rem .95rem .35rem .65rem!important;margin:0 6px 8px 0!important}
label[data-baseweb="radio"]>div:first-child{background-color:#fff!important;border-color:#2F6FD0!important}
label[data-baseweb="radio"],label[data-baseweb="radio"] *{color:#0D1530!important;font-weight:600}
label[data-baseweb="radio"]:has(input:checked){background:#0D1530!important;border-color:#0D1530!important}
label[data-baseweb="radio"]:has(input:checked),label[data-baseweb="radio"]:has(input:checked) p{color:#FACF35!important}
label[data-baseweb="checkbox"]>span:first-child{background:#fff!important;border:2px solid #2F6FD0!important}
label[data-baseweb="checkbox"]:has(input:checked)>span:first-child{background:#2F6FD0!important}
label[data-baseweb="checkbox"] *{color:#0D1530!important}
[role="tablist"]{background:#fff!important;border:0!important;border-radius:18px;padding:6px;gap:6px;box-shadow:0 16px 32px -16px rgba(80,50,0,.55)}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}
[role="tab"]{border-radius:13px!important;padding:.75rem 1.4rem!important;background:transparent!important}
[role="tab"],[role="tab"] *{color:#0D1530!important;font-size:1.08rem!important;font-weight:800!important}
[role="tab"]:hover{background:#FFF0B8!important}
[role="tab"][aria-selected="true"]{background:#0D1530!important}
[role="tab"][aria-selected="true"] *{color:#FACF35!important}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] *,.stApp [data-testid="stExpander"] summary *,.stApp [data-testid="stExpanderDetails"] p:not([class]),.stApp [data-testid="stExpanderDetails"] li{color:#0D1530!important}
.stApp [data-testid="stExpander"]{background:#fff!important;border:0!important;border-radius:16px!important}
.stApp [data-testid^="stBaseButton"]{background:#fff!important;border:1.5px solid #0D1530!important;border-radius:999px!important}
.stApp [data-testid^="stBaseButton"] *{color:#0D1530!important;font-weight:700!important}
.stApp [data-testid^="stBaseButton"]:hover{background:#FACF35!important}
.perks i{background:#fff;box-shadow:0 8px 16px -6px rgba(80,50,0,.45)}
'''
MANTAP_CSS = MANTAP_CSS.replace("</style>", MANTAP_FIX + "</style>")
TEMA_NAMA = {"mantap": "☀️ Kuning Mantap", "neon": "🌌 Neon Aurora", "editorial": "📰 Showroom Editorial", "cockpit": "🕹️ Cockpit HUD", "brutal": "🟨 Neo-Brutal"}
TEMA_CSS = {"mantap": MANTAP_CSS, "neon": "", "editorial": '<style>\n@import url(\'https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=Inter:wght@400;600;700&display=swap\');\n:root{--ink:#17140F;--txt:#2B2720;--mut:#6B6458;--dim:#8A8372;--line:rgba(23,20,15,.2);--acc:#C2410C;--cy:#C2410C;--vi:#17140F;--pk:#C2410C;--grad:linear-gradient(90deg,#17140F,#C2410C);--surf:#FFFDF8;--card:#FFFDF8;--track:rgba(23,20,15,.1);--hi:#FBE4D3;--onacc:#fff}\n.stApp,.stApp p,.stApp label,.stApp button,.stApp input,.stApp table{font-family:\'Inter\',sans-serif}\n.stApp{background:radial-gradient(rgba(23,20,15,.09) 1px,transparent 1px) 0 0/22px 22px,#F2ECDF!important}\n.hero h1,.rp,.stat b,.kpi b,.pc b,.dc b,.vt,.ins-h,.sum b,.pv{font-family:\'Fraunces\',Georgia,serif!important;letter-spacing:-.02em}\n.hero,.res,.vcard,.plan,.stop,.card2,.proj,.how,.pc,.stat,.kpi div,.sum,.st-key-spec{border-radius:3px!important;border:1.5px solid var(--ink)!important}\n.hero{border-width:2px!important;box-shadow:10px 10px 0 var(--acc)}\n.res,.vcard{box-shadow:6px 6px 0 var(--ink)}\n.pc:hover,.stat:hover{box-shadow:4px 4px 0 var(--acc)}\n.rp .num{font-weight:800}\n.badge{border-radius:0!important;border:1.5px solid var(--ink)!important;letter-spacing:.2em}\n.stat,.pc,.card2,.proj,.how,.kpi div,.sum,.chip,.badge,.st-key-spec{background:var(--surf)!important}\n.res,.vcard.ok,.vcard.low,.vcard.high,.plan,.stop,.hero{background:var(--card)!important}\n.pc.mid{background:var(--hi)!important}\n.gauge,.hb .ht,.meter i{background:var(--track)}\n.hero h1,.stat b,.how .st b,.pc b,.pv,.dc b,.vt,.sum b,.ins-h,.kpi b,.card2 h5,.proj h5,.hb .hv,.stop h3,.proj>p b,.badge{color:var(--ink)!important}\n.hero:before,.hero:after,.res:after{display:none}\n.hero .art,.res .carart{filter:brightness(0)}\n.hero .road{background:repeating-linear-gradient(90deg,var(--ink) 0 34px,transparent 34px 68px)}\n.needle{background:linear-gradient(var(--ink),rgba(0,0,0,.2));filter:none}\n.hub{background:var(--ink);box-shadow:0 0 0 5px var(--line)}\n.chip b,.note a,.ft a{color:var(--acc)}\n.tb td{color:var(--txt);border-bottom:1px solid var(--line)}\n.al.warn{background:#FFF1C9;border-color:#C98A00;color:#6B4400}\n.al.info{background:#E3F0F7;border-color:#2B7A9B;color:#0B4A5E}\n.stApp input,.stApp div[data-baseweb="select"] *,.stApp summary,.stApp summary *{color:var(--ink)!important}\ndiv[data-baseweb="select"]>div,div[data-baseweb="base-input"]{background:#fff!important}\ndiv[data-baseweb="popover"] ul,div[data-baseweb="popover"] li{background:#fff!important;color:#111!important}\ndiv[data-baseweb="popover"] li *{color:#111!important}\ndiv[data-baseweb="tab-list"]{background:var(--surf)}\nbutton[data-baseweb="tab"][aria-selected="true"],div[role="radiogroup"]>label:has(input:checked){background:var(--ink)!important;color:var(--onacc)}\ndiv[role="radiogroup"]>label{background:#fff}\n.stButton>button,.stDownloadButton>button{background:#fff;color:var(--ink)}\n</style>', "cockpit": '<style>\n@import url(\'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&display=swap\');\n:root{--line:rgba(255,176,0,.3);--txt:#E4F7E8;--mut:#86B293;--dim:#5C8068;--cy:#FFB000;--vi:#FFB000;--pk:#3CFF9A;--grad:linear-gradient(90deg,#FFB000,#3CFF9A)}\n.stApp,.stApp p,.stApp label,.stApp button,.stApp input,.stApp h1,.stApp h2,.stApp h3,.stApp table,.rp,.stat b,.kpi b,.pc b,.dc b,.sum b,.pv{font-family:\'JetBrains Mono\',monospace!important}\n.stApp{background:repeating-linear-gradient(0deg,rgba(60,255,154,.045) 0 1px,transparent 1px 3px),radial-gradient(900px 500px at 50% -10%,rgba(255,176,0,.16),transparent 60%),#050A08!important}\n.hero,.res,.vcard,.plan,.stop,.card2,.proj,.how,.pc,.stat,.kpi div,.sum,.st-key-spec,.chip,.badge{border-radius:2px!important;border:1px solid var(--line)!important;box-shadow:inset 0 0 26px rgba(255,176,0,.05)}\n.hero:before{background:radial-gradient(circle,rgba(60,255,154,.22),transparent 65%)}\n.hero h1{text-transform:uppercase;letter-spacing:-.01em!important}\n.hero .art,.res .carart{filter:sepia(1) saturate(5) hue-rotate(5deg) drop-shadow(0 0 14px rgba(255,176,0,.6))}\n.rp .num{filter:drop-shadow(0 0 14px rgba(255,176,0,.5))}\n.stat:hover,.pc:hover{border-color:#FFB000!important;box-shadow:0 0 22px rgba(255,176,0,.3)}\nbutton[data-baseweb="tab"][aria-selected="true"],div[role="radiogroup"]>label:has(input:checked){background:#FFB000!important;color:#000}\ndiv[role="radiogroup"]>label:has(input:checked) *{color:#000!important}\n.step i,.how .st i{color:#000}\n</style>', "brutal": '<style>\n@import url(\'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=DM+Mono:wght@500&display=swap\');\n:root{--ink:#000;--txt:#111;--mut:#333;--dim:#555;--line:#000;--acc:#3D5AFE;--cy:#3D5AFE;--vi:#3D5AFE;--pk:#FF5CA8;--grad:linear-gradient(90deg,#FF5CA8,#3D5AFE);--surf:#fff;--card:#fff;--track:#e2e2e2;--hi:#FFE14D;--onacc:#fff}\n.stApp,.stApp p,.stApp label,.stApp button,.stApp input,.stApp h1,.stApp table{font-family:\'Space Grotesk\',sans-serif}\n.stApp{background:#FFD93D!important}\n.rp,.stat b,.kpi b,.pc b,.pv,.sum b{font-family:\'DM Mono\',monospace!important}\n.hero,.res,.vcard,.plan,.stop,.card2,.proj,.how,.pc,.stat,.kpi div,.sum,.st-key-spec,.chip,.badge{border-radius:0!important;border:3px solid #000!important;box-shadow:6px 6px 0 #000}\n.pc,.stat,.kpi div,.sum,.chip{border-width:2px!important;box-shadow:3px 3px 0 #000}\n.hero{background:#FF8FC7!important}\n.hero h1{font-weight:800!important;text-transform:uppercase}\n.res,.vcard{transition:transform .15s,box-shadow .15s}\n.res:hover,.vcard:hover,.card2:hover,.proj:hover{transform:translate(-3px,-3px);box-shadow:9px 9px 0 #000}\n.stat:hover,.pc:hover,.kpi div:hover{transform:translate(-2px,-2px);box-shadow:5px 5px 0 #000}\n.badge{background:#FFD93D!important}\n.pb,.rb i{background:#3D5AFE!important;border:2px solid #000}\n.pcol.now .pb{background:#FF5CA8!important}\n.band{background:#FF5CA8;border:2px solid #000}\n.stat,.pc,.card2,.proj,.how,.kpi div,.sum,.chip,.badge,.st-key-spec{background:var(--surf)!important}\n.res,.vcard.ok,.vcard.low,.vcard.high,.plan,.stop,.hero{background:var(--card)!important}\n.pc.mid{background:var(--hi)!important}\n.gauge,.hb .ht,.meter i{background:var(--track)}\n.hero h1,.stat b,.how .st b,.pc b,.pv,.dc b,.vt,.sum b,.ins-h,.kpi b,.card2 h5,.proj h5,.hb .hv,.stop h3,.proj>p b,.badge{color:var(--ink)!important}\n.hero:before,.hero:after,.res:after{display:none}\n.hero .art,.res .carart{filter:brightness(0)}\n.hero .road{background:repeating-linear-gradient(90deg,var(--ink) 0 34px,transparent 34px 68px)}\n.needle{background:linear-gradient(var(--ink),rgba(0,0,0,.2));filter:none}\n.hub{background:var(--ink);box-shadow:0 0 0 5px var(--line)}\n.chip b,.note a,.ft a{color:var(--acc)}\n.tb td{color:var(--txt);border-bottom:1px solid var(--line)}\n.al.warn{background:#FFF1C9;border-color:#C98A00;color:#6B4400}\n.al.info{background:#E3F0F7;border-color:#2B7A9B;color:#0B4A5E}\n.stApp input,.stApp div[data-baseweb="select"] *,.stApp summary,.stApp summary *{color:var(--ink)!important}\ndiv[data-baseweb="select"]>div,div[data-baseweb="base-input"]{background:#fff!important}\ndiv[data-baseweb="popover"] ul,div[data-baseweb="popover"] li{background:#fff!important;color:#111!important}\ndiv[data-baseweb="popover"] li *{color:#111!important}\ndiv[data-baseweb="tab-list"]{background:var(--surf)}\nbutton[data-baseweb="tab"][aria-selected="true"],div[role="radiogroup"]>label:has(input:checked){background:var(--ink)!important;color:var(--onacc)}\ndiv[role="radiogroup"]>label{background:#fff}\n.stButton>button,.stDownloadButton>button{background:#fff;color:var(--ink)}\n</style>'}

FUSION_CSS = r"""<style>
:root{--ink:#0D1530;--yel:#FACF35;--blu:#2F6FD0;--cy:#FACF35;--vi:#FFB81C;--pk:#FF8A6B;--grad:linear-gradient(90deg,#FACF35,#FF9F43 55%,#F472B6)}
.stApp{background:radial-gradient(900px 520px at 6% -6%,rgba(250,207,53,.20),transparent 60%),radial-gradient(1000px 760px at 82% 22%,rgba(47,111,208,.50),transparent 66%),radial-gradient(700px 420px at 50% 112%,rgba(244,114,182,.12),transparent 60%),linear-gradient(180deg,#101A3D,#0B1330)!important}
.block-container{max-width:1240px}
.how{display:none}

.fh .badge{color:#FFF3C4;border-color:rgba(250,207,53,.38);background:rgba(250,207,53,.08)}
.fh h1{font-size:clamp(2.5rem,5.2vw,4.4rem)!important;line-height:1.04!important;font-weight:800!important;letter-spacing:-.04em!important;color:#fff!important;margin:.9rem 0 .7rem!important;padding:0!important}
.fh h1 em{font-style:normal;background:#FACF35;color:#0D1530;padding:0 .16em;border-radius:.2em;display:inline-block;transform:rotate(-1.8deg);box-shadow:0 14px 34px -10px rgba(250,207,53,.65);animation:wob 3s ease-in-out infinite alternate}
.fh p.sub{max-width:560px;color:var(--mut);font-size:1.05rem;line-height:1.65;margin:0 0 .4rem}
.fh .stage{position:relative;height:270px;margin:.6rem 0 1.3rem}
.fh .stage:before{content:"";position:absolute;left:4%;right:4%;bottom:10px;height:75%;background:radial-gradient(ellipse at 50% 85%,rgba(250,207,53,.38),transparent 68%);filter:blur(8px)}
.fh .stage img.car{position:absolute;left:3%;bottom:16px;width:80%;animation:drive 3s ease-in-out infinite alternate;filter:drop-shadow(0 22px 18px rgba(0,0,0,.55)) drop-shadow(0 0 22px rgba(250,207,53,.25))}
.fh .stage .road{position:absolute;left:0;right:0;bottom:8px;height:4px;border-radius:4px;background:repeating-linear-gradient(90deg,#FACF35 0 34px,transparent 34px 68px);animation:road .9s linear infinite;opacity:.8}
.fh .coin{position:absolute;width:76px;height:76px;border-radius:50%;background:radial-gradient(circle at 32% 28%,#FFF6B8,#F7B500 55%,#B97800);border:4px solid #E09A00;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1.2rem;color:#8A5500;box-shadow:0 14px 28px rgba(0,0,0,.45),0 0 26px rgba(250,207,53,.35);animation:bob 3s ease-in-out infinite}
.fh .coin.c1{right:4%;top:2%}.fh .coin.c2{left:0;top:26%;width:54px;height:54px;font-size:.9rem;animation-delay:-1s}.fh .coin.c3{right:16%;bottom:0;width:96px;height:96px;font-size:1.5rem;animation-delay:-2s}
.fh .stats{max-width:none}
.fh .stat{background:rgba(255,255,255,.06);border-color:rgba(250,207,53,.22)}
.fh .stat b{color:#FACF35}
.fh .tick{margin-top:1.1rem}
.fh .tick span:before{color:#FACF35}

.st-key-spec{position:relative;background:#F7F9FF!important;border:0!important;border-radius:26px!important;padding:1.3rem 1.4rem 1rem 3.3rem!important;box-shadow:0 0 0 1px rgba(250,207,53,.5),0 0 90px rgba(120,160,255,.28),0 40px 80px -30px rgba(250,207,53,.30),0 30px 60px -20px rgba(0,0,0,.45)}
.st-key-spec:before{content:"";position:absolute;left:27px;top:56px;bottom:50px;width:4px;border-radius:4px;background:repeating-linear-gradient(180deg,#2F6FD0 0 10px,transparent 10px 20px);animation:flow 1s linear infinite}
.st-key-spec:after{content:"\2713";position:absolute;left:14px;bottom:18px;width:30px;height:30px;border-radius:50%;border:3px solid #2F6FD0;color:#2F6FD0;background:#fff;display:flex;align-items:center;justify-content:center;font-weight:800}
.st-key-spec .ct{position:relative;font-size:1.05rem;color:#0D1530;text-transform:none;letter-spacing:0;font-weight:800}
.st-key-spec .ct:before{content:"\2605";position:absolute;left:-2.6rem;top:-5px;width:30px;height:30px;border-radius:50%;background:#2F6FD0;color:#fff;display:flex;align-items:center;justify-content:center;font-size:.8rem}
.st-key-spec [data-testid="stWidgetLabel"] p{color:#0D1530!important;font-weight:700!important;font-size:.82rem!important}
.st-key-spec [data-baseweb="select"] *,.st-key-spec [data-baseweb="input"] *,.st-key-spec [data-baseweb="base-input"] *,.st-key-spec [data-testid="stNumberInputContainer"],.st-key-spec [data-testid="stNumberInputContainer"] *{background-color:#fff!important;color:#0D1530!important;-webkit-text-fill-color:#0D1530!important}
.st-key-spec [data-baseweb="select"]>div,.st-key-spec [data-baseweb="input"]{border:1.5px solid #C9D1E8!important;border-radius:12px!important}
.st-key-spec [data-baseweb="select"]:focus-within>div,.st-key-spec [data-baseweb="input"]:focus-within{border-color:#2F6FD0!important;box-shadow:0 0 0 3px rgba(47,111,208,.22)!important}
.st-key-spec input:disabled{opacity:.45}
.st-key-spec [data-baseweb="select"],.st-key-spec [data-baseweb="select"]>div,.st-key-spec [data-baseweb="select"]>div>div,.st-key-spec [data-testid="stSelectbox"] [data-baseweb="select"] div{background-color:#fff!important;color:#0D1530!important;-webkit-text-fill-color:#0D1530!important}
.st-key-spec [data-baseweb="select"] svg{fill:#0D1530!important;color:#0D1530!important}
.st-key-spec [data-baseweb="select"]>div,.st-key-spec [data-baseweb="input"],.st-key-spec [data-baseweb="base-input"]{border:1.5px solid #C9D1E8!important;border-radius:12px!important}
.st-key-spec label[data-baseweb="radio"]{background:#fff!important;border:1.5px solid #C9D1E8!important;border-radius:999px!important;padding:.35rem .95rem .35rem .65rem!important;margin:0 6px 8px 0!important;cursor:pointer}
.st-key-spec label[data-baseweb="radio"],.st-key-spec label[data-baseweb="radio"] *{color:#0D1530!important;-webkit-text-fill-color:#0D1530!important;font-weight:600!important}
.st-key-spec label[data-baseweb="radio"]>div:first-child{background-color:#fff!important;border-color:#2F6FD0!important}
.st-key-spec label[data-baseweb="radio"]:has(input:checked){background:#FACF35!important;border-color:#0D1530!important}
.st-key-spec label[data-baseweb="radio"]:has(input:checked)>div:first-child{background-color:#2F6FD0!important;border-color:#2F6FD0!important}
.st-key-spec label[data-baseweb="checkbox"]>:first-child{background-color:#fff!important;border:2px solid #2F6FD0!important;border-radius:6px!important}
.st-key-spec label[data-baseweb="checkbox"]:has(input:checked)>:first-child{background-color:#2F6FD0!important}
.st-key-spec label[data-baseweb="checkbox"] *{color:#0D1530!important;-webkit-text-fill-color:#0D1530!important}
.st-key-spec [data-testid="stExpander"]{background:#F3F5FB!important;border:0!important;border-radius:16px!important}
.st-key-spec [data-testid="stExpander"] summary *{color:#0D1530!important;font-weight:700}
.st-key-spec [data-testid^="stBaseButton"]{background:#fff!important;border:1.5px solid #0D1530!important;border-radius:999px!important}
.st-key-spec [data-testid^="stBaseButton"] *{color:#0D1530!important;font-weight:700!important}
.st-key-spec [data-testid^="stBaseButton"]:hover{background:#FACF35!important}
div[data-baseweb="popover"] *{background-color:#fff!important;color:#0D1530!important}
div[data-baseweb="popover"] li:hover,div[data-baseweb="popover"] li:hover *,div[data-baseweb="popover"] [aria-selected="true"],div[data-baseweb="popover"] [aria-selected="true"] *{background-color:#FFF0B8!important}

.st-key-spec [data-testid="stSelectbox"] div{background-color:#fff!important}
.st-key-spec [data-testid="stSelectbox"] div:has(> input){border:1px solid #B8BDC9!important;border-radius:8px!important;min-height:48px;align-items:center}
.st-key-spec [data-testid="stSelectbox"] div:has(> input):focus-within{border-color:#2F6FD0!important;box-shadow:0 0 0 3px rgba(47,111,208,.2)!important}
.st-key-spec [data-testid="stSelectbox"] input{color:#0D1530!important;-webkit-text-fill-color:#0D1530!important;background:#fff!important;font-size:1rem!important;font-weight:500}
.st-key-spec [data-testid="stSelectbox"] svg{fill:#0D1530!important;color:#0D1530!important}
.st-key-spec [data-testid="stSelectbox"] button{background:transparent!important}
.st-key-spec [data-testid="stNumberInputContainer"]{border:1px solid #B8BDC9!important;border-radius:8px!important;min-height:48px}
.st-key-spec [data-testid="stNumberInputContainer"] input{font-size:1rem!important}

[data-testid="stSelectboxVirtualDropdown"],[data-testid="stSelectboxVirtualDropdown"] [role="listbox"],[data-testid="stSelectboxVirtualDropdown"] div{background-color:#fff!important}
[data-testid="stSelectboxVirtualDropdown"]{border:0!important;border-radius:10px!important;box-shadow:0 12px 36px rgba(0,0,0,.28)!important;overflow:hidden}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]{padding:.8rem 1.15rem!important;font-size:1rem!important;color:#222!important;background-color:#fff!important}
[data-testid="stSelectboxVirtualDropdown"] [role="option"] *{color:#222!important;background-color:transparent!important;font-size:1rem!important}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-hovered="true"],[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"]{background-color:#F1F3F8!important}
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"]{background-color:#EAF2FD!important}
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] *{font-weight:700!important}

.st-key-spec [data-testid="stRadioOption"]{background:#fff!important;border:1px solid #B8BDC9!important;border-radius:10px!important;padding:.5rem 1rem .5rem .75rem!important;margin:0 6px 8px 0!important;cursor:pointer}
.st-key-spec [data-testid="stRadioOption"],.st-key-spec [data-testid="stRadioOption"] *{color:#0D1530!important;-webkit-text-fill-color:#0D1530!important;font-weight:600!important}
.st-key-spec [data-testid="stRadioOption"]>div>div:first-of-type{background-color:#fff!important;border:2px solid #2F6FD0!important}
.st-key-spec [data-testid="stRadioOption"][data-selected="true"],.st-key-spec [data-testid="stRadioOption"]:has(input:checked){background:#EAF2FD!important;border:2px solid #2F6FD0!important}
.st-key-spec [data-testid="stRadioOption"][data-selected="true"]>div>div:first-of-type,.st-key-spec [data-testid="stRadioOption"]:has(input:checked)>div>div:first-of-type{background-color:#2F6FD0!important}
.st-key-spec [data-testid="stRadioOption"][data-selected="true"]>div>div:first-of-type *,.st-key-spec [data-testid="stRadioOption"]:has(input:checked)>div>div:first-of-type *{background-color:#fff!important}

.st-key-spec [data-testid="stCheckbox"] *{color:#0D1530!important;-webkit-text-fill-color:#0D1530!important}
.st-key-spec [data-testid="stCheckbox"]>div>:first-child{background-color:#fff!important;border:2px solid #2F6FD0!important;border-radius:6px!important}
.st-key-spec [data-testid="stCheckbox"]>div>:first-child:has(svg){background-color:#2F6FD0!important}
.st-key-spec [data-testid="stCheckbox"] :has(> svg){background-color:#2F6FD0!important;border-color:#2F6FD0!important}
.st-key-spec [data-testid="stCheckbox"] svg polyline{stroke:#fff!important;fill:none!important}

[role="tablist"]{background:rgba(255,255,255,.06)!important;border:1px solid var(--line)!important;border-radius:18px;padding:6px;gap:6px}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}
[role="tab"]{border-radius:13px!important;padding:.7rem 1.3rem!important;background:transparent!important}
[role="tab"],[role="tab"] *{color:#E9EDF7!important;font-size:1.02rem!important;font-weight:800!important}
[role="tab"]:hover{background:rgba(250,207,53,.14)!important}
[role="tab"][aria-selected="true"]{background:#FACF35!important}
[role="tab"][aria-selected="true"] *{color:#0D1530!important}
div[role="radiogroup"]>label:has(input:checked){background:#FACF35!important;border-color:#FACF35!important}
div[role="radiogroup"]>label:has(input:checked) *{color:#0D1530!important}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#FACF35;color:#fff;box-shadow:0 0 0 3px rgba(250,207,53,.2)}

.res{border:1px solid rgba(250,207,53,.42);background:linear-gradient(160deg,rgba(250,207,53,.16),rgba(47,111,208,.10) 60%,rgba(255,255,255,.03));box-shadow:0 26px 70px -26px rgba(250,207,53,.45)}
.res .carart{filter:sepia(1) saturate(4) hue-rotate(5deg) drop-shadow(0 0 14px rgba(250,207,53,.4))}
.pc.mid{border-color:rgba(250,207,53,.65);background:rgba(250,207,53,.14)}
.pb{background:linear-gradient(180deg,rgba(250,207,53,.7),rgba(250,207,53,.14))}
.rb i{background:linear-gradient(180deg,#FACF35,rgba(255,159,67,.45))}
.mk{border-color:#FACF35;box-shadow:0 0 0 6px rgba(250,207,53,.25)}
.mk.pin{border-color:#FB7185;box-shadow:0 0 0 6px rgba(251,113,133,.25)}
.how .st i{color:#0D1530}
.ft a,.note a,.chip b{color:#FACF35}

@keyframes bob{50%{transform:translateY(-16px) rotate(9deg)}}
@keyframes drive{from{transform:translateX(-12px) rotate(-.5deg)}to{transform:translateX(16px) rotate(.5deg)}}
@keyframes wob{to{transform:rotate(1.2deg) scale(1.03)}}
@keyframes flow{to{background-position:0 20px}}
@media (max-width:760px){.fh .stage{height:200px}.fh .coin.c3{display:none}.st-key-spec{padding-left:3rem!important}}
</style>"""
TEMA_CSS["fusion"] = FUSION_CSS
TEMA_NAMA["fusion"] = "✨ Fusion (Neon + Mantap)"

_BODI = {
    "Sedan": ("M18 76 L18 66 Q18 60 28 58 L58 53 L82 33 Q86 29 94 29 L148 29 Q156 29 162 35 L178 53 L212 59 Q224 61 224 69 L224 76 L197 76 A15 15 0 0 0 167 76 L81 76 A15 15 0 0 0 51 76 Z",
              "M72 52 L88 35 L124 35 L124 52 Z M130 35 L148 35 L166 52 L130 52 Z"),
    "Hatchback": ("M20 76 L20 62 Q20 57 28 56 L68 52 L76 33 Q79 29 87 29 L146 29 Q154 29 160 35 L176 53 L208 59 Q218 61 218 69 L218 76 L197 76 A15 15 0 0 0 167 76 L81 76 A15 15 0 0 0 51 76 Z",
                  "M74 51 L82 35 L122 35 L122 51 Z M128 35 L148 35 L164 51 L128 51 Z"),
    "SUV": ("M22 73 L22 46 Q22 38 30 35 L52 29 Q58 22 70 22 L160 22 Q168 22 173 28 L190 44 L214 49 Q225 51 227 59 L227 73 L197 73 A15 15 0 0 0 167 73 L81 73 A15 15 0 0 0 51 73 Z",
            "M34 40 L54 33 Q60 27 70 27 L110 27 L110 40 Z M116 27 L158 27 Q164 27 168 32 L184 42 L184 40 L116 40 Z"),
    "MPV": ("M16 74 L16 52 Q16 45 24 43 L40 41 L55 24 Q59 20 67 20 L168 20 Q179 20 185 28 L203 46 L220 52 Q228 56 228 66 L228 74 L197 74 A15 15 0 0 0 167 74 L81 74 A15 15 0 0 0 51 74 Z",
            "M44 41 L58 27 L98 27 L98 41 Z M104 27 L168 27 Q175 27 179 33 L193 45 L104 45 Z"),
    "Coupe": ("M16 76 L16 67 Q16 62 26 60 L62 55 L96 36 Q102 32 112 32 L146 32 Q156 32 162 38 L192 55 L216 59 Q226 62 226 69 L226 76 L197 76 A15 15 0 0 0 167 76 L81 76 A15 15 0 0 0 51 76 Z",
              "M70 55 L100 38 L130 38 L130 55 Z M136 38 L150 38 L178 55 L136 55 Z"),
    "Pickup": ("M12 76 L12 56 L96 56 L96 40 L118 28 L168 28 Q176 28 180 33 L196 52 L218 56 Q228 58 228 66 L228 76 L197 76 A15 15 0 0 0 167 76 L81 76 A15 15 0 0 0 51 76 Z",
               "M102 54 L102 43 L120 33 L142 33 L142 54 Z M148 33 L168 33 L184 54 L148 54 Z"),
    "Van": ("M14 76 L14 38 Q14 30 24 30 L176 30 Q192 30 198 40 L218 56 Q228 60 228 67 L228 76 L197 76 A15 15 0 0 0 167 76 L81 76 A15 15 0 0 0 51 76 Z",
            "M24 36 L72 36 L72 54 L24 54 Z M80 36 L172 36 Q184 36 188 44 L200 54 L80 54 Z"),
}

def svg_mobil(bodi, tebal=2.6):
    badan, kaca = _BODI.get(bodi, _BODI["Sedan"])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 100">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#22D3EE"/>'
        '<stop offset=".55" stop-color="#8B5CF6"/><stop offset="1" stop-color="#F472B6"/></linearGradient></defs>'
        f'<g fill="none" stroke="url(#g)" stroke-width="{tebal}" stroke-linejoin="round" stroke-linecap="round">'
        f'<path d="{badan}" fill="#8B5CF6" fill-opacity=".12"/><path d="{kaca}" stroke-width="{tebal * .6}"/>'
        '<circle cx="66" cy="76" r="13" fill="#0B1226"/><circle cx="66" cy="76" r="5"/>'
        '<circle cx="182" cy="76" r="13" fill="#0B1226"/><circle cx="182" cy="76" r="5"/>'
        '</g></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

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

def persen_str(x):
    return "0%" if abs(x) < 0.5 else f"{x:+.0f}%"

@st.cache_resource
def muat_model():
    return joblib.load(PATH_MODEL)

def buat_baris(info, merek, model, tahun, km, mesin_l, bahan_bakar, bodi, transmisi, umur=None):
    lengkap = f"{merek} {model}".strip()
    model_g = lengkap if lengkap in info["model_dikenal"] else f"{merek} LAINNYA"
    if umur is None:
        umur = max(info["tahun_acuan"] - tahun, 0)
    return {"merek": merek, "model_g": model_g, "bahan_bakar": bahan_bakar, "bodi": bodi,
            "transmisi": transmisi, "sumber": info["sumber_prediksi"], "umur": umur, "km": km,
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

def proyeksi_nilai(paket, merek, model, tahun, km, mesin_l, bahan_bakar, bodi, transmisi, tahun_ke_depan=5):
    """Estimasi harga 0..N tahun ke depan: umur bertambah, kilometer bertambah dengan laju pemakaian yang sama.
    Nilai dibuat tidak naik seiring umur (penghalusan), karena model pohon kadang sedikit bergelombang."""
    info = paket["info"]
    umur0 = max(info["tahun_acuan"] - tahun, 0)
    laju = km / max(umur0, 1) if not np.isnan(km) else np.nan
    baris = [buat_baris(info, merek, model, tahun, (km + laju * t) if not np.isnan(laju) else np.nan,
                        mesin_l, bahan_bakar, bodi, transmisi, umur=umur0 + t)
             for t in range(tahun_ke_depan + 1)]
    nilai = paket["est"].predict(pd.DataFrame(baris))
    return [float(v) for v in np.minimum.accumulate(nilai)]

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

def ringkasan_teks(spek, hasil):
    return (
        "ESTIMASI HARGA MOBIL BEKAS\n"
        "==========================\n"
        f"Mobil     : {spek}\n"
        f"Estimasi  : {rp(hasil['estimasi'])}\n"
        f"Rentang   : {rp(hasil['rendah'])} sampai {rp(hasil['tinggi'])}\n"
        f"Keyakinan : {hasil['keyakinan']}\n\n"
        "Perkiraan statistik dari iklan marketplace, bukan penilaian resmi atau jaminan harga.\n"
    )

WARNA_KEYAKINAN = {"Tinggi": "#34D399", "Sedang": "#FBBF24", "Rendah": "#FB923C"}
SEGMEN = {"Tinggi": 3, "Sedang": 2, "Rendah": 1}
PENJELASAN = {
    "Tinggi": "Model ini punya banyak data ({n} iklan), jadi estimasinya paling bisa dipercaya.",
    "Sedang": "Data model ini lumayan ({n} iklan), tapi belum banyak. Rentang harga bisa lebih lebar.",
    "Rendah": "Model ini jarang atau tidak ada di data, jadi ini estimasi kasar berdasarkan mobil sejenis. "
              "Anggap sebagai gambaran awal, bukan patokan harga.",
}

def svg_mobil_warna(bodi, warna="#E8402A", garis="#0D1530"):
    badan, kaca = _BODI.get(bodi, _BODI["MPV"])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 100"><defs>'
        f'<linearGradient id="b" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{warna}"/>'
        '<stop offset="1" stop-color="#7A1408"/></linearGradient></defs>'
        f'<path d="{badan}" fill="url(#b)" stroke="{garis}" stroke-width="2" stroke-linejoin="round"/>'
        f'<path d="{kaca}" fill="#BFE3F5" stroke="{garis}" stroke-width="1.4" stroke-linejoin="round"/>'
        '<rect x="208" y="60" width="13" height="5" rx="2" fill="#FFE27A"/><rect x="14" y="60" width="7" height="5" rx="2" fill="#FF4D4D"/>'
        '<circle cx="66" cy="76" r="13" fill="#111"/><circle cx="66" cy="76" r="6" fill="#C9CED8"/>'
        '<circle cx="182" cy="76" r="13" fill="#111"/><circle cx="182" cy="76" r="6" fill="#C9CED8"/></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

def html_hero_mantap(bodi):
    return f"""
    <div class="mh">
      <h1>Berapa harga <em>mantap</em> mobilmu?</h1>
      <p class="sub">Cepat, gratis, dan bebas repot. Tanpa nomor HP.</p>
      <div class="stage"><div class="coin c1">Rp</div><div class="coin c2">Rp</div><div class="coin c3">Rp</div>
        <img class="car" src="{svg_mobil_warna(bodi)}" alt=""><div class="road"></div></div>
      <div class="perks"><div><i>⚡</i>Estimasi<br>instan</div><div><i>📊</i>Rentang<br>harga wajar</div>
        <div><i>📉</i>Proyeksi nilai<br>5 tahun</div><div><i>🔒</i>Tanpa data<br>pribadi</div></div>
    </div>
    """

def html_hero_fusion(info, bodi):
    total = sum(info["jumlah_per_merek"].values())
    mape = info["skor_cv"]["Model dikenal, km diketahui"]["MAPE (%)"]
    tick = '<div class="tick"><div>' + "".join(f"<span>{m}</span>" for m in info["merek_didukung"]) * 2 + "</div></div>"
    return f"""
    <div class="fh">
      <div class="badge"><i></i> Machine learning · Gradient Boosting</div>
      <h1>Berapa harga <em>mantap</em> <span class="grad">mobil bekasmu</span>?</h1>
      <p class="sub">Isi spesifikasi, lalu dapatkan estimasi harga, rentang wajar, dan proyeksi nilai 5 tahun.
      Dilatih dari ribuan iklan mobil bekas nyata di Indonesia. Gratis, tanpa nomor HP.</p>
      <div class="stage"><div class="coin c1">Rp</div><div class="coin c2">Rp</div><div class="coin c3">Rp</div>
        <img class="car" src="{svg_mobil_warna(bodi, garis='#F8FAFF')}" alt=""><div class="road"></div></div>
      <div class="stats">
        <div class="stat"><b>{angka_id(total)}</b><span>iklan untuk melatih</span></div>
        <div class="stat"><b>{len(info['merek_didukung'])} merek</b><span>{len(info['model_dikenal'])} model dikenal</span></div>
        <div class="stat"><b>±{mape:.0f}%</b><span>rata-rata meleset (model populer)</span></div>
      </div>
      {tick}
    </div>
    """

def html_hero(info):
    total = sum(info["jumlah_per_merek"].values())
    mape = info["skor_cv"]["Model dikenal, km diketahui"]["MAPE (%)"]
    tick = '<div class="tick"><div>' + "".join(f"<span>{m}</span>" for m in info["merek_didukung"]) * 2 + "</div></div>"
    return f"""
    <div class="hero">
      <img class="art" src="{svg_mobil('Coupe', 2.2)}" alt="">
      <div class="road"></div>
      <div class="badge"><i></i> Machine learning · Gradient Boosting</div>
      <h1>Berapa harga <span class="grad">mobil bekasmu</span> yang pantas?</h1>
      <p class="sub">Isi spesifikasi mobil, lalu dapatkan estimasi harga, rentang wajar, proyeksi nilai ke depan, dan
      tingkat keyakinan model. Dilatih dari ribuan iklan mobil bekas nyata di Indonesia.</p>
      <div class="stats">
        <div class="stat"><b>{angka_id(total)}</b><span>iklan untuk melatih</span></div>
        <div class="stat"><b>{len(info['merek_didukung'])} merek</b><span>{len(info['model_dikenal'])} model dikenal</span></div>
        <div class="stat"><b>±{mape:.0f}%</b><span>rata-rata meleset (model populer)</span></div>
      </div>
      {tick}
    </div>
    """

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

def html_hasil_penjual(hasil, bodi):
    angka, satuan = rp_bagian(hasil["estimasi"])
    if satuan == "juta":
        nilai = f'<span class="num cnt" style="--to:{int(round(hasil["estimasi"] / 1e6))}" aria-label="{angka}"></span>'
    else:
        nilai = f'<span class="num">{angka}</span>'
    return f"""
    <div class="res">
      <img class="carart" src="{svg_mobil(bodi)}" alt="">
      <div class="rl">Estimasi harga pasang</div>
      <div class="rp"><span class="cur">Rp</span>{nilai}<span class="unit">{satuan}</span></div>
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

def html_proyeksi(nilai, acuan):
    v0 = nilai[0]
    kolom = ""
    for t, v in enumerate(nilai):
        tinggi = 16 + 118 * (v / v0)
        label = "Sekarang" if t == 0 else f"+{t} th"
        turun = "" if t == 0 else persen_str((v / v0 - 1) * 100)
        kolom += (f'<div class="pcol{" now" if t == 0 else ""}"><span class="pv">{angka_id(v / 1e6)}</span>'
                  f'<div class="pb" style="height:{tinggi:.0f}px"></div><span class="py">{label}</span>'
                  f'<span class="pd">{turun}</span></div>')
    kartu = ""
    for t in (1, 3, 5):
        if t < len(nilai):
            kartu += (f'<div class="pc"><em>{t} tahun lagi</em><b>{rp(nilai[t])}</b>'
                      f'<span>hilang {rp(v0 - nilai[t])}</span></div>')
    rata = (v0 - nilai[-1]) / (len(nilai) - 1)
    return f"""
    <div class="proj">
      <h5>Nilai mobil ke depan</h5>
      <p>Estimasi model (juta Rp) bila mobil dipakai dengan pola yang sama: umur bertambah dan kilometer naik seperti
      biasa. Rata-rata penyusutan sekitar <b style="color:#fff">{rp(rata)}</b> per tahun. Gambaran pola, bukan jaminan.</p>
      <div class="pbars">{kolom}</div>
      <div class="pst">{kartu}</div>
    </div>
    """

def sudut_dial(harga, rendah, tinggi):
    if harga < rendah:
        p = max(0.0, (harga / rendah - 0.6) / 0.4) / 3
    elif harga > tinggi:
        p = 2 / 3 + min(1.0, (harga / tinggi - 1) / 0.4) / 3
    else:
        p = 1 / 3 + (harga - rendah) / (tinggi - rendah) / 3
    return -90 + p * 180

def html_vonis(hasil, harga):
    est = hasil["estimasi"]
    selisih = (harga - est) / est * 100
    if harga < hasil["rendah"]:
        tipe, judul, pendek = "low", "Di bawah rentang wajar", "Murah"
        teks = (f"{persen_str(selisih)} dari estimasi. Bisa jadi murah, atau ada kondisi yang perlu dicek "
                "(surat, bekas tabrak atau banjir, kilometer diturunkan).")
    elif harga > hasil["tinggi"]:
        tipe, judul, pendek = "high", "Di atas rentang wajar", "Mahal"
        teks = (f"{persen_str(selisih)} dari estimasi. Layak ditawar, atau cek apakah ada kelebihan khusus "
                "yang tidak ikut dihitung model.")
    else:
        tipe, judul, pendek = "ok", "Harga wajar", "Wajar"
        teks = f"Masih dalam rentang wajar ({persen_str(selisih)} dari estimasi)."
    if harga > est * 1.02:
        teks += f" Titik tawar yang masuk akal: sekitar {rp(est)} (selisih {rp(harga - est)})."
    sudut = sudut_dial(harga, hasil["rendah"], hasil["tinggi"])
    return f"""
    <div class="vcard {tipe}">
      <div><div class="dial" style="--a:{sudut:.1f}deg"><div class="arc"></div><div class="needle"></div><div class="hub"></div>
      </div>
      <div class="dlab"><span>Murah</span><span>Wajar</span><span>Mahal</span></div>
      <div class="dc"><b>{pendek}</b><span>{persen_str(selisih)} dibanding estimasi</span></div></div>
      <div><div class="vt">{judul}</div><div class="vx">{teks}</div></div>
    </div>
    <div class="res" style="margin-top:.9rem;padding:1.1rem 1.3rem">
      <div class="rl">Harga pasang (kuning) dibanding estimasi (ungu)</div>
      {html_gauge(hasil['rendah'], est, hasil['tinggi'], harga)}
    </div>
    <div class="sum"><span>Estimasi model <b>{rp(est)}</b></span><span>Rentang <b>{rp(hasil['rendah'])} - {rp(hasil['tinggi'])}</b></span><span>Keyakinan <b>{hasil['keyakinan']}</b></span></div>
    """

def html_rencana(nilai, harga, tahun_pakai):
    rasio = nilai[tahun_pakai] / nilai[0]
    jual_kembali = harga * rasio
    susut = harga - jual_kembali
    return f"""
    <div class="plan">
      <div class="rl">Kalau dipakai {tahun_pakai} tahun</div>
      <div class="pst" style="margin-top:.7rem">
        <div class="pc"><em>Nilai jual kembali</em><b>{rp(jual_kembali)}</b><span>perkiraan</span></div>
        <div class="pc"><em>Penyusutan total</em><b>{rp(susut)}</b><span>{persen_str((rasio - 1) * 100)}</span></div>
        <div class="pc mid"><em>Per tahun</em><b>{rp(susut / tahun_pakai)}</b><span>biaya penyusutan</span></div>
      </div>
      <p>Dihitung dari harga pasang dikali pola penurunan nilai model untuk mobil ini. Belum termasuk bensin, pajak,
      servis, dan asuransi.</p>
    </div>
    """

def html_insight():
    ins = INSIGHT
    tautan_nb = ""
    if LINK_GITHUB:
        tautan_nb = (f' <a href="{LINK_GITHUB.rstrip("/")}/blob/main/notebook/03_insight_bisnis.ipynb" '
                     'target="_blank">Lihat analisis lengkap di GitHub &rarr;</a>')
    ret = ins["retensi"]
    kolom = "".join(
        f'<div class="rb"><em>{v if (i + 1) in (1, 3, 5, 7, 10, 12, 15) else ""}</em>'
        f'<i style="height:{10 + 1.15 * v:.0f}px;animation-delay:{i * 0.03:.2f}s"></i><span>{i + 1}</span></div>'
        for i, v in enumerate(ret))
    mk = max(v for _, v in ins["merek"])
    baris_merek = "".join(
        f'<div class="hb{" hot" if v > 10 else ""}"><span class="hl">{n}</span><div class="ht"><div class="hf" style="width:{v / mk * 100:.0f}%"></div></div>'
        f'<span class="hv">{str(v).replace(".", ",")}%</span></div>' for n, v in ins["merek"])
    mr = max(v for _, v in ins["rupiah"])
    baris_rp = "".join(
        f'<div class="hb{" hot" if v > 60 else ""}"><span class="hl">{n}</span><div class="ht"><div class="hf" style="width:{v / mr * 100:.0f}%"></div></div>'
        f'<span class="hv">Rp {str(v).replace(".", ",")} jt</span></div>' for n, v in ins["rupiah"])
    return f"""
    <div class="ins-h">Apa yang menentukan harga mobil bekas?</div>
    <p class="ins-s">Ringkasan dari analisis {angka_id(ins['iklan'])} iklan marketplace (notebook 03). Perbandingan dilakukan di dalam model
    mobil yang sama, jadi bukan campuran antar merek.</p>
    <div class="kpi">
      <div><b>-{100 - ret[4]}%</b><span>harga turun dalam 5 tahun pertama</span></div>
      <div><b>-{100 - ret[9]}%</b><span>harga turun dalam 10 tahun</span></div>
      <div><b>±0,9%</b><span>efek +10.000 km setelah umur dikontrol</span></div>
    </div>
    <div class="card2"><h5>Harga menurut umur (umur 1 tahun = 100)</h5>
      <p>Penurunan ±6-9% per tahun sampai umur 12, lalu makin curam (11-14% per tahun).</p>
      <div class="rbars">{kolom}</div></div>
    <div class="card2"><h5>Seberapa cepat harga turun per merek</h5>
      <p>Median laju penurunan harga per tahun, umur 1-8 tahun.</p>{baris_merek}</div>
    <div class="card2"><h5>Biaya penyusutan dalam rupiah (mobil umur 5 tahun)</h5>
      <p>Perkiraan kehilangan nilai per tahun. Penyusutan C-Class sekitar 5-6 kali Avanza.</p>{baris_rp}</div>
    <div class="card2"><h5>Premium untuk varian (model dan umur sama)</h5>
      <div class="chips"><span class="chip">Automatic <b>+7%</b></span><span class="chip">Diesel <b>+11% s.d. +27%</b></span>
      <span class="chip">Hybrid penuh <b>+25%</b></span><span class="chip">Mild hybrid <b>+3%</b></span>
      <span class="chip">Listrik vs bensin (umur 1-4 th) <b>sama, ±5% per tahun</b></span></div></div>
    <p class="note">Semua angka adalah asosiasi pada harga iklan (bukan harga transaksi) dan bukan sebab-akibat. Detail dan
    keterbatasan ada di notebook 03_insight_bisnis.{tautan_nb}</p>
    """

def html_cara_kerja(info):
    total = sum(info["jumlah_per_merek"].values())
    return f"""
    <div class="how">
      <div class="ct">Cara kerjanya</div>
      <div class="st"><i>1</i><div><b>Data nyata</b><span>{angka_id(total)} iklan dari OLX, Mobil123, dan Carsome, dibersihkan dari duplikat dan data janggal.</span></div></div>
      <div class="st"><i>2</i><div><b>Model belajar pola</b><span>Gradient Boosting mempelajari hubungan harga dengan model, umur, kilometer, dan spesifikasi.</span></div></div>
      <div class="st"><i>3</i><div><b>Hasil yang jujur</b><span>Estimasi plus rentang wajar dan tingkat keyakinan. Merek tanpa data cukup ditolak, bukan ditebak.</span></div></div>
    </div>
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
    koma = lambda x: f"{x:.1f}".replace(".", ",")
    baris = ""
    for kunci, label in nama.items():
        s = skor[kunci]
        baris += (f"<tr><td>{label}</td><td>{koma(s['MAE (juta Rp)'])}</td><td>{koma(s['MAPE (%)'])}%</td>"
                  f"<td>{koma(s['Median error (%)'])}%</td></tr>")
    return (f'<table class="tb"><tr><th>Skenario uji</th><th>Meleset (juta Rp)</th><th>Meleset (%)</th>'
            f'<th>Median (%)</th></tr>{baris}</table>')

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

try:
    tema = st.query_params.get("tema", "fusion")
except Exception:
    tema = "fusion"
if tema not in TEMA_CSS:
    tema = "fusion"
tampil(CSS)
tampil(TEMA_CSS[tema])
if tema not in ("mantap", "fusion"):
    tampil(html_hero(info))

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
st.session_state.setdefault("rencana", 3)

def pakai_contoh(nama):
    st.session_state.update(CONTOH[nama])
    st.session_state.pop("harga_pasang", None)

def kartu(nama):
    try:
        return st.container(border=True, key=nama)
    except TypeError:
        return st.container(border=True)

if tema in ("mantap", "fusion"):
    kol_hero, kiri = st.columns([6, 5], gap="large")
    with kol_hero:
        if tema == "fusion":
            tampil(html_hero_fusion(info, st.session_state.get("bodi", "MPV")))
        else:
            tampil(html_hero_mantap(st.session_state.get("bodi", "MPV")))
    kanan = st.container()
else:
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
                bodi = st.selectbox("Tipe bodi", bodi_opsi, key="bodi")

            transmisi = st.radio("Transmisi", pilihan["transmisi"], horizontal=True, key="transmisi")
            bahan_bakar = st.radio("Bahan bakar", pilihan["bahan_bakar"], horizontal=True, key="bahan_bakar")

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

    tampil(html_cara_kerja(info))

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
        nilai = proyeksi_nilai(paket, merek, model, tahun, km, mesin_l, bahan_bakar, bodi, transmisi)
        tab_jual, tab_beli, tab_insight = st.tabs(["💰 Saya penjual", "🛒 Saya pembeli", "📈 Insight pasar"])

        with tab_jual:
            tampil(html_hasil_penjual(hasil, bodi))
            tampil(html_proyeksi(nilai, info["tahun_acuan"]))
            st.caption("Rentang dihitung dari seberapa jauh tebakan model meleset pada data uji, dan mencakup sekitar "
                       "4 dari 5 harga iklan mobil serupa.")
            nama_mobil = (f"{merek} {model or '(model lain)'} {tahun}, {transmisi}, {bahan_bakar}, {bodi}, "
                          f"{'tanpa data km' if km_na else angka_id(km) + ' km'}"
                          + ("" if listrik else f", {mesin_l:.1f} L"))
            st.download_button("Unduh ringkasan estimasi (.txt)", data=ringkasan_teks(nama_mobil, hasil),
                               file_name="estimasi-harga-mobil.txt", mime="text/plain")

        with tab_beli:
            st.session_state.setdefault("harga_pasang", int(round(hasil["estimasi"] / 1e6)))
            harga_pasang = st.number_input("Harga yang dipasang penjual (juta Rp)", min_value=10, max_value=20000,
                                           step=5, key="harga_pasang")
            harga = harga_pasang * 1e6
            tampil(html_vonis(hasil, harga))
            pakai = st.slider("Rencana dipakai berapa tahun?", 1, 5, key="rencana")
            tampil(html_rencana(nilai, harga, pakai))
            st.caption("Ini bantuan pembanding, bukan pengganti cek fisik dan surat mobil.")

        with tab_insight:
            tampil(html_insight())

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
            "- Proyeksi nilai ke depan adalah pola dari data saat ini, bukan ramalan harga masa depan.\n"
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