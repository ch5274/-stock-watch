import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Watch v6", page_icon="📡", layout="wide")

# 모바일 폭을 최대한 활용
st.markdown("""
<style>
.block-container {padding-top:1rem; padding-left:.65rem; padding-right:.65rem; max-width:1400px;}
div[data-testid="stMetric"] {padding: .15rem;}
div[data-testid="stMetricValue"] {font-size:1.15rem;}
@media (max-width: 700px) {
  .block-container {padding-left:.35rem; padding-right:.35rem;}
  h1 {font-size:1.65rem !important;} h2 {font-size:1.3rem !important;}
  div[data-testid="stMetricValue"] {font-size:1rem;}
}
</style>
""", unsafe_allow_html=True)

DEFAULTS={"전진건설로봇":"079900.KS","페니트리움바이오":"187660.KQ","현대바이오사이언스":"048410.KQ","Moderna":"MRNA"}
if "watch" not in st.session_state: st.session_state.watch=DEFAULTS.copy()

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False).mean(); ad=dn.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+au/ad.replace(0,np.nan))

def indicators(d):
    x=d.copy()
    for n in (5,10,20,60,120,200): x[f"MA{n}"]=x.Close.rolling(n).mean()
    x["RSI"]=rsi(x.Close)
    lo=x.RSI.rolling(14).min(); hi=x.RSI.rolling(14).max()
    x["STOCH_RSI"]=100*(x.RSI-lo)/(hi-lo).replace(0,np.nan)
    x["MACD"]=ema(x.Close,12)-ema(x.Close,26); x["MACD_SIG"]=ema(x.MACD,9); x["MACD_H"]=x.MACD-x.MACD_SIG
    x["VOL20"]=x.Volume.rolling(20).mean(); x["VOL_RATIO"]=x.Volume/x.VOL20
    pc=x.Close.shift(1); tr=pd.concat([(x.High-x.Low),(x.High-pc).abs(),(x.Low-pc).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.rolling(14).mean(); x["ATR_PCT"]=100*x.ATR/x.Close
    mid=x.Close.rolling(20).mean(); sd=x.Close.rolling(20).std()
    x["BB_MID"]=mid; x["BB_UP"]=mid+2*sd; x["BB_LO"]=mid-2*sd
    up=x.High.diff(); dn=-x.Low.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    atr=tr.ewm(alpha=1/14,adjust=False).mean()
    pdi=100*pd.Series(plus,index=x.index).ewm(alpha=1/14,adjust=False).mean()/atr
    mdi=100*pd.Series(minus,index=x.index).ewm(alpha=1/14,adjust=False).mean()/atr
    x["PDI"]=pdi; x["MDI"]=mdi
    x["ADX"]=(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)).ewm(alpha=1/14,adjust=False).mean()
    tp=(x.High+x.Low+x.Close)/3
    direction=tp.diff()
    pos=(tp*x.Volume).where(direction>0,0).rolling(14).sum()
    neg=(tp*x.Volume).where(direction<0,0).rolling(14).sum()
    x["MFI"]=100-100/(1+pos/neg.replace(0,np.nan))
    x["OBV"]=(np.sign(x.Close.diff()).fillna(0)*x.Volume).cumsum()
    md=(tp-tp.rolling(20).mean()).abs().rolling(20).mean()
    x["CCI"]=(tp-tp.rolling(20).mean())/(0.015*md.replace(0,np.nan))
    hh=x.High.rolling(14).max(); ll=x.Low.rolling(14).min()
    x["WILLR"]=-100*(hh-x.Close)/(hh-ll).replace(0,np.nan)
    x["ROC"]=100*x.Close.pct_change(12)
    x["HH20"]=x.High.shift(1).rolling(20).max()
    return x

@st.cache_data(ttl=120)
def daily(t):
    d=yf.download(t,period="3y",interval="1d",auto_adjust=False,progress=False,threads=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])

@st.cache_data(ttl=60)
def intraday(t,iv):
    period="5d" if iv in ("5m","15m","30m") else "1mo"
    d=yf.download(t,period=period,interval=iv,auto_adjust=False,progress=False,threads=False,prepost=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])

def timeframe(d,k):
    if k=="일": return indicators(d)
    rule="W-FRI" if k=="주" else "ME"
    return indicators(d.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna())

def divergence(x,n=20):
    z=x.tail(n)
    if len(z)<n or z.RSI.isna().all(): return "없음"
    h=n//2
    if z.Close.iloc[h:].min()<z.Close.iloc[:h].min() and z.RSI.iloc[h:].min()>z.RSI.iloc[:h].min(): return "상승 후보"
    if z.Close.iloc[h:].max()>z.Close.iloc[:h].max() and z.RSI.iloc[h:].max()<z.RSI.iloc[:h].max(): return "하락 후보"
    return "없음"

def analyze(x):
    x=x.dropna(subset=["Close"]); a=x.iloc[-1]; b=x.iloc[-2]; p=float(a.Close)
    def f(k,default=0): return float(a[k]) if k in a and pd.notna(a[k]) else default
    ma={n:f(f"MA{n}",p) for n in (5,10,20,60,120,200)}
    rv=f("RSI",50); vr=f("VOL_RATIO",0)
    sup=float(x.Low.tail(20).min()); res=float(x.High.iloc[-21:-1].max())
    rebound=p>float(b.Close) and float(a.Close)>float(a.Open)
    s1=(p<=sup*1.04 or ma[20]*.98<=p<=ma[20]*1.03 or ma[60]*.98<=p<=ma[60]*1.03) and rebound and vr>=1.10
    broke=bool((x.Close.iloc[-7:-1]>x.HH20.iloc[-7:-1]).fillna(False).any())
    s2=broke and res*.985<=p<=res*1.035 and p>=float(b.Close)
    drop=(x.Close.tail(8).min()/x.Close.tail(8).max()-1)<=-.07
    fade=x.Volume.tail(3).mean()<x.Volume.iloc[-8:-3].mean() if len(x)>=8 else False
    s3=drop and fade and rebound and vr>=1.10
    s4=p>ma[20] and float(b.Close)<=float(x.MA20.iloc[-2])*1.02 and rebound and vr>=1.05
    tight=(x.High.tail(10).max()/x.Low.tail(10).min()-1)<.07
    s5=tight and p>res and vr>=1.4
    spread=max(ma[5],ma[10],ma[20])-min(ma[5],ma[10],ma[20])
    s6=(spread/p<.025) and p>ma[5]>ma[20] and vr>=1.15
    s7=p>res and vr>=1.3
    patterns=[i for i,v in enumerate((s1,s2,s3,s4,s5,s6,s7),1) if v]
    trend="상승" if p>ma[20]>ma[60] else ("하락" if p<ma[20]<ma[60] else "혼조")
    score=50+(12 if trend=="상승" else -12 if trend=="하락" else 0)
    score+=7 if ma[5]>ma[10]>ma[20] else 0
    score+=7 if f("MACD")>f("MACD_SIG") and f("MACD_H")>float(x.MACD_H.iloc[-2]) else -3
    score+=5 if 45<=rv<=68 else -6 if rv>=75 else 2 if rv<35 else 0
    score+=6 if f("PDI")>f("MDI") and f("ADX")>=20 else 0
    score+=5 if f("MFI",50)>50 else -2
    score+=5 if x.OBV.iloc[-1]>x.OBV.rolling(20).mean().iloc[-1] else -2
    score+=7 if vr>=1.2 and rebound else -5 if vr>=1.5 and not rebound else 0
    score+=10 if patterns else 0
    score=max(0,min(100,score))
    if patterns: verdict="🟢 추가매수 확인"
    elif rv>=72 or p>ma[20]*1.10: verdict="🟠 추격 금지"
    elif p<sup*.97 or (p<ma[60] and f("MACD")<f("MACD_SIG")): verdict="🔴 손상/위험"
    elif score>=65: verdict="🔵 돌파 확인 대기"
    else: verdict="⚪ 관망"
    return dict(p=p,ma=ma,rsi=rv,vr=vr,sup=sup,res=res,trend=trend,score=score,verdict=verdict,
                macd="강세" if f("MACD")>f("MACD_SIG") else "약세",adx=f("ADX"),mfi=f("MFI",50),
                stoch=f("STOCH_RSI",50),cci=f("CCI"),willr=f("WILLR",-50),roc=f("ROC"),
                atr=f("ATR_PCT"),div=divergence(x),patterns=patterns,invalid=min(sup,ma[60])*.97)

def intra_analyze(d):
    if d is None or len(d)<25: return None
    x=d.copy()
    idx=pd.DatetimeIndex(x.index)
    keys=idx.date
    tp=(x.High+x.Low+x.Close)/3
    x["VWAP"]=(tp*x.Volume).groupby(keys).cumsum()/x.Volume.groupby(keys).cumsum().replace(0,np.nan)
    x["EMA9"]=ema(x.Close,9); x["EMA20"]=ema(x.Close,20); x["RSI"]=rsi(x.Close)
    a=x.iloc[-1]; p=float(a.Close); vw=float(a.VWAP) if pd.notna(a.VWAP) else p
    prior=x.tail(21).iloc[:-1]
    sup=float(prior.Low.min()); res=float(prior.High.max())
    vbase=float(prior.Volume.mean()); vr=float(a.Volume/vbase) if vbase else 0
    reclaim=(float(x.Close.iloc[-2])<=float(x.VWAP.iloc[-2]) and p>vw) if pd.notna(x.VWAP.iloc[-2]) else False
    bullish=p>vw and p>float(a.EMA9)>float(a.EMA20)
    breakout=p>res and vr>=1.3
    pullback=abs(p-vw)/p<=.006 and p>=vw and float(a.Close)>float(a.Open)
    if breakout: sig="🟢 장중 돌파"
    elif reclaim and vr>=1.1: sig="🟢 VWAP 재돌파"
    elif pullback: sig="🔵 VWAP 눌림 확인"
    elif bullish: sig="🔵 단기 상승 유지"
    elif p<vw and p<float(a.EMA20): sig="🔴 단기 약세"
    else: sig="⚪ 대기"
    return dict(x=x,p=p,vwap=vw,rsi=float(a.RSI) if pd.notna(a.RSI) else 50,vr=vr,sup=sup,res=res,sig=sig)


def intraday_chart(q, bars=60):
    z=q["x"].tail(bars).copy()
    # Price panel + volume panel. Range is explicitly limited to visible bars.
    lo=float(min(z.Low.min(),z.VWAP.min(),z.EMA9.min(),z.EMA20.min()))
    hi=float(max(z.High.max(),z.VWAP.max(),z.EMA9.max(),z.EMA20.max()))
    span=max(hi-lo, hi*.002)
    ymin=lo-span*.06
    ymax=hi+span*.06

    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
                      row_heights=[0.78,0.22],vertical_spacing=0.025)
    fig.add_trace(go.Candlestick(
        x=z.index,open=z.Open,high=z.High,low=z.Low,close=z.Close,
        name="가격",showlegend=False
    ),row=1,col=1)
    fig.add_trace(go.Scatter(x=z.index,y=z.VWAP,name="VWAP",mode="lines",line={"width":2}),row=1,col=1)
    fig.add_trace(go.Scatter(x=z.index,y=z.EMA9,name="EMA9",mode="lines",line={"width":1.3}),row=1,col=1)
    fig.add_trace(go.Scatter(x=z.index,y=z.EMA20,name="EMA20",mode="lines",line={"width":1.3}),row=1,col=1)
    fig.add_trace(go.Bar(x=z.index,y=z.Volume,name="거래량",showlegend=False),row=2,col=1)

    fig.update_yaxes(range=[ymin,ymax],row=1,col=1,fixedrange=False)
    fig.update_xaxes(rangeslider_visible=False,row=1,col=1)
    fig.update_xaxes(rangeslider_visible=False,row=2,col=1)
    fig.update_layout(
        height=610,
        margin={"l":4,"r":4,"t":30,"b":4},
        legend={"orientation":"h","yanchor":"bottom","y":1.01,"xanchor":"left","x":0},
        hovermode="x unified",
        dragmode="pan"
    )
    return fig

def money(v,t): return f"{v:,.0f}원" if t.endswith((".KS",".KQ")) else f"${v:,.2f}"
PAT={1:"지지반등+거래량",2:"돌파후 눌림",3:"급락후 회복",4:"20일선 눌림",5:"박스 돌파",6:"이평 수렴→확산",7:"전고점 돌파"}

st.title("📡 Stock Watch v7")
st.caption("v4 고급분석 + v5 장중분석 유지 · 모바일 확대 캔들차트")

with st.sidebar:
    st.header("관심종목")
    nm=st.text_input("이름",placeholder="NVIDIA"); tk=st.text_input("티커",placeholder="NVDA / 005930.KS").strip().upper()
    if st.button("➕ 추가",use_container_width=True) and tk:
        st.session_state.watch[nm.strip() or tk]=tk; st.rerun()
    if st.session_state.watch:
        dele=st.selectbox("삭제",["선택 안 함"]+list(st.session_state.watch))
        if st.button("➖ 삭제",use_container_width=True) and dele!="선택 안 함":
            st.session_state.watch.pop(dele,None); st.rerun()

rows=[]; raw={}
for n,t in st.session_state.watch.items():
    try:
        d=daily(t); raw[t]=d; a=analyze(timeframe(d,"일"))
        rows.append([n,t,a["verdict"],a["score"],a["trend"],round(a["rsi"],1),round(a["vr"],2)])
    except Exception: rows.append([n,t,"조회 실패",0,"-",np.nan,np.nan])

st.subheader("전체 감시판")
for r in rows:
    with st.container(border=True):
        st.markdown(f"**{r[0]}** · `{r[1]}`")
        c=st.columns(3); c[0].metric("판정",r[2]); c[1].metric("점수",r[3]); c[2].metric("추세",r[4])
        st.caption(f"RSI {r[5]} · 거래량 {r[6]}×")

if st.session_state.watch:
    name=st.selectbox("상세 종목",list(st.session_state.watch)); t=st.session_state.watch[name]
    d=raw.get(t,daily(t)); frames={k:timeframe(d,k) for k in ("일","주","월")}
    A={k:analyze(v) for k,v in frames.items() if len(v)>25}; a=A["일"]

    st.subheader(f"{name} · {t}")
    c=st.columns(3); c[0].metric("가격",money(a["p"],t)); c[1].metric("점수",f'{a["score"]}/100'); c[2].metric("판정",a["verdict"])
    st.write("**월→주→일:** "+" · ".join(f"{k} {q['trend']}({q['score']})" for k,q in reversed(list(A.items()))))
    st.write(f"지지 **{money(a['sup'],t)}** · 저항 **{money(a['res'],t)}** · 무효 **{money(a['invalid'],t)}**")
    if a["patterns"]: st.success("패턴: "+" / ".join(PAT[i] for i in a["patterns"]))

    with st.expander("🧠 고급 기술분석",expanded=False):
        st.write(f"RSI {a['rsi']:.1f} · Stoch RSI {a['stoch']:.1f} · MACD {a['macd']} · ADX {a['adx']:.1f}")
        st.write(f"MFI {a['mfi']:.1f} · CCI {a['cci']:.1f} · Williams %R {a['willr']:.1f} · ROC {a['roc']:.1f}% · ATR {a['atr']:.2f}%")
        st.write(f"RSI 다이버전스: **{a['div']}**")
        st.line_chart(frames["일"][["Close","MA5","MA10","MA20","MA60","MA120","MA200"]].tail(220),height=430)

    st.subheader("⚡ 장중 진입 타이밍")
    choice=st.segmented_control("분봉 선택",["5분","15분","30분","60분"],default="15분")
    iv={"5분":"5m","15분":"15m","30분":"30m","60분":"60m"}[choice]
    bars_label=st.segmented_control("확대 범위",["30봉","60봉","120봉"],default="60봉")
    bars={"30봉":30,"60봉":60,"120봉":120}[bars_label]
    try:
        q=intra_analyze(intraday(t,iv))
        if q:
            st.markdown(f"### {q['sig']}")
            c=st.columns(2)
            c[0].metric("가격",money(q["p"],t)); c[1].metric("VWAP",money(q["vwap"],t))
            c=st.columns(2)
            c[0].metric("RSI",f'{q["rsi"]:.1f}'); c[1].metric("거래량",f'{q["vr"]:.2f}×')
            st.write(f"단기 지지 **{money(q['sup'],t)}** · 단기 저항 **{money(q['res'],t)}**")
            st.plotly_chart(intraday_chart(q,bars),use_container_width=True,config={"displayModeBar":False,"scrollZoom":True})
            st.caption("캔들 + VWAP + EMA9/20 + 거래량 · Y축은 선택한 봉의 실제 가격 범위에 맞춰 자동 확대")
        else: st.info("선택한 분봉 데이터가 부족합니다.")
    except Exception as e: st.warning(f"장중 데이터 조회 실패: {e}")

    with st.expander("📆 일·주·월 큰 추세"):
        k=st.radio("큰 차트",["일","주","월"],horizontal=True)
        st.line_chart(frames[k][["Close","MA20","MA60"]].tail(160 if k=="일" else 90),height=430)

st.warning("현재 장중 데이터는 Yahoo 기반이라 지연/누락될 수 있습니다. 실제 체결·외국인/기관/프로그램·공매도·알림·주문은 실시간 API 연결 단계에서 추가합니다.")
