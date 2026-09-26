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

@st.cache_data(ttl=300, show_spinner=False)
def search_symbols(query):
    q=(query or "").strip()
    if len(q)<1:
        return []
    try:
        results=yf.Search(q,max_results=12,news_count=0,lists_count=0,
                          include_research=False,enable_fuzzy_query=True).quotes
        out=[]
        for r in results:
            symbol=str(r.get("symbol","")).strip()
            if not symbol:
                continue
            qt=str(r.get("quoteType","")).upper()
            if qt not in ("EQUITY","ETF","MUTUALFUND","INDEX"):
                continue
            name=r.get("shortname") or r.get("longname") or r.get("name") or symbol
            exch=r.get("exchDisp") or r.get("exchange") or ""
            typ={"EQUITY":"주식","ETF":"ETF","MUTUALFUND":"펀드","INDEX":"지수"}.get(qt,qt)
            out.append({"symbol":symbol,"name":str(name),"exchange":str(exch),"type":typ})
        return out
    except Exception:
        return []

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


def decision_text(a, intra_map, t):
    good=[]
    weak=[]
    for label,q in intra_map.items():
        if not q: continue
        if "🟢" in q["sig"] or "상승 유지" in q["sig"] or "눌림 확인" in q["sig"]:
            good.append(label)
        if "🔴" in q["sig"]:
            weak.append(label)

    confirm=max(a["res"], a["p"]*1.003)
    chase=max(confirm*1.02, a["p"]*1.025)
    invalid=a["invalid"]
    entry_lo=max(a["sup"], a["p"]*.985)
    entry_hi=min(confirm, a["p"]*1.005)

    reasons=[]
    reasons.append(f"일봉 {a['trend']} · 기술점수 {a['score']}/100")
    if a["patterns"]:
        reasons.append("진입패턴: "+", ".join(PAT[i] for i in a["patterns"]))
    if good: reasons.append("장중 강세/확인: "+", ".join(good))
    if weak: reasons.append("장중 약세: "+", ".join(weak))
    reasons.append(f"거래량 {a['vr']:.2f}배 · RSI {a['rsi']:.1f}")

    if a["trend"]=="상승" and len(good)>=2 and not weak:
        verdict="🟢 추가매수 관찰"
        action=f"{money(confirm,t)} 돌파/지지 확인 시 분할 접근"
        confidence=min(92,a["score"]+10+len(good)*3)
    elif a["verdict"].startswith("🟠"):
        verdict="🟠 추격 금지"
        action="현재 가격 추격보다 눌림 또는 지지 확인 대기"
        confidence=max(55,a["score"])
    elif a["trend"]=="하락" or len(weak)>=2:
        verdict="🔴 진입 보류"
        action=f"{money(invalid,t)} 부근 위험관리, 추세 회복 전 신규진입 보류"
        confidence=min(90,60+len(weak)*7)
    elif len(good)>=1 or a["score"]>=65:
        verdict="🔵 확인 대기"
        action=f"{money(confirm,t)} 돌파와 거래량 유지 확인"
        confidence=min(85,a["score"]+5)
    else:
        verdict="⚪ 관망"
        action="조건이 모일 때까지 대기"
        confidence=max(45,a["score"])

    return {
        "verdict":verdict,"action":action,"confidence":int(confidence),
        "entry_lo":entry_lo,"entry_hi":entry_hi,"confirm":confirm,
        "chase":chase,"invalid":invalid,"reasons":reasons,
        "good":good,"weak":weak
    }

def money(v,t): return f"{v:,.0f}원" if t.endswith((".KS",".KQ")) else f"${v:,.2f}"
PAT={1:"지지반등+거래량",2:"돌파후 눌림",3:"급락후 회복",4:"20일선 눌림",5:"박스 돌파",6:"이평 수렴→확산",7:"전고점 돌파"}

st.title("📡 Stock Watch v9 · 검색형 관심종목")
st.caption("회사명/티커 검색 → ⭐ 추가 → 자동 판독 · 기존 기술분석 엔진 유지")

with st.sidebar:
    st.header("⭐ 관심종목 관리")
    st.caption("회사명이나 티커를 검색하고 결과를 눌러 추가하세요.")

    query=st.text_input("🔎 종목 검색",placeholder="예: Moderna, NVDA, 삼성전자")
    results=search_symbols(query) if query.strip() else []

    if query.strip():
        if results:
            labels=[
                f"{r['name']}  |  {r['symbol']}  |  {r['exchange']}  |  {r['type']}"
                for r in results
            ]
            chosen_label=st.selectbox("검색 결과",labels)
            chosen=results[labels.index(chosen_label)]
            already=chosen["symbol"] in st.session_state.watch.values()
            if already:
                st.info("이미 관심종목에 들어 있습니다.")
            elif st.button("⭐ 이 종목 관심종목에 추가",use_container_width=True):
                # Duplicate names are disambiguated by symbol.
                display=chosen["name"]
                if display in st.session_state.watch and st.session_state.watch[display]!=chosen["symbol"]:
                    display=f"{display} ({chosen['symbol']})"
                st.session_state.watch[display]=chosen["symbol"]
                st.rerun()
        else:
            st.warning("검색 결과가 없습니다. 한글 검색이 안 잡히면 영문 회사명이나 티커로 검색해 주세요.")

    st.divider()
    st.markdown("**현재 관심종목**")
    if st.session_state.watch:
        remove_name=st.selectbox("관심종목에서 빼기",["선택 안 함"]+list(st.session_state.watch.keys()))
        if remove_name!="선택 안 함":
            st.caption(f"{remove_name} · {st.session_state.watch[remove_name]}")
        if st.button("🗑️ 선택 종목 빼기",use_container_width=True,disabled=remove_name=="선택 안 함"):
            st.session_state.watch.pop(remove_name,None)
            st.rerun()
    else:
        st.caption("등록된 관심종목이 없습니다.")

    with st.expander("티커 직접 추가 · 검색이 안 될 때"):
        nm=st.text_input("표시 이름",placeholder="예: 삼성전자")
        tk=st.text_input("Yahoo 티커",placeholder="예: 005930.KS / NVDA").strip().upper()
        if st.button("직접 추가",use_container_width=True) and tk:
            st.session_state.watch[nm.strip() or tk]=tk
            st.rerun()

rows=[]; raw={}
for n,t in st.session_state.watch.items():
    try:
        d=daily(t); raw[t]=d; a=analyze(timeframe(d,"일"))
        rows.append([n,t,a["verdict"],a["score"],a["trend"],round(a["rsi"],1),round(a["vr"],2)])
    except Exception: rows.append([n,t,"조회 실패",0,"-",np.nan,np.nan])

st.subheader("⭐ 관심종목")
st.caption("관심종목은 왼쪽 메뉴에서 언제든 추가·삭제할 수 있습니다.")
for r in rows:
    with st.container(border=True):
        st.markdown(f"### {r[0]}")
        st.caption(f"{r[1]} · {r[4]} 추세")
        st.markdown(f"**{r[2]}** · 종합점수 **{r[3]}/100**")
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

    st.subheader("🎯 지금 어떻게 볼까")
    intra_map={}
    for label,iv in {"5분":"5m","15분":"15m","30분":"30m","60분":"60m"}.items():
        try:
            intra_map[label]=intra_analyze(intraday(t,iv))
        except Exception:
            intra_map[label]=None

    dec=decision_text(a,intra_map,t)
    if dec["verdict"].startswith("🟢"):
        st.success(f"### {dec['verdict']}")
    elif dec["verdict"].startswith("🔴"):
        st.error(f"### {dec['verdict']}")
    elif dec["verdict"].startswith("🟠"):
        st.warning(f"### {dec['verdict']}")
    else:
        st.info(f"### {dec['verdict']}")

    st.markdown(f"**행동:** {dec['action']}")
    st.progress(dec["confidence"]/100, text=f"판단 신뢰도 {dec['confidence']}/100")

    c=st.columns(2)
    c[0].metric("관찰/진입 후보",f"{money(dec['entry_lo'],t)} ~ {money(dec['entry_hi'],t)}")
    c[1].metric("확인 가격",money(dec["confirm"],t))
    c=st.columns(2)
    c[0].metric("추격 주의",money(dec["chase"],t))
    c[1].metric("무효/위험",money(dec["invalid"],t))

    st.markdown("**왜 이렇게 봤나**")
    for reason in dec["reasons"]:
        st.write("• "+reason)

    st.markdown("**5·15·30·60분 자동 판독**")
    cols=st.columns(2)
    for i,label in enumerate(["5분","15분","30분","60분"]):
        q=intra_map.get(label)
        with cols[i%2]:
            if q:
                st.markdown(f"**{label}** {q['sig']}")
                st.caption(f"VWAP {money(q['vwap'],t)} · RSI {q['rsi']:.0f} · 거래량 {q['vr']:.2f}×")
            else:
                st.markdown(f"**{label}** 데이터 부족")

    with st.expander("📊 차트가 필요할 때만 보기",expanded=False):
        choice=st.segmented_control("분봉",["5분","15분","30분","60분"],default="15분")
        bars_label=st.segmented_control("범위",["30봉","60봉","120봉"],default="60봉")
        q=intra_map.get(choice)
        if q:
            bars={"30봉":30,"60봉":60,"120봉":120}[bars_label]
            st.plotly_chart(intraday_chart(q,bars),use_container_width=True,
                            config={"displayModeBar":False,"scrollZoom":True})
            st.caption("캔들 + VWAP + EMA9/20 + 거래량")
        else:
            st.info("선택한 분봉 데이터가 부족합니다.")

    with st.expander("📆 일·주·월 큰 추세"):
        k=st.radio("큰 차트",["일","주","월"],horizontal=True)
        st.line_chart(frames[k][["Close","MA20","MA60"]].tail(160 if k=="일" else 90),height=430)

st.warning("현재 장중 데이터는 Yahoo 기반이라 지연/누락될 수 있습니다. 실제 체결·외국인/기관/프로그램·공매도·알림·주문은 실시간 API 연결 단계에서 추가합니다.")
