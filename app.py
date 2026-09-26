import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Stock Watch v5", page_icon="⚡", layout="wide")
DEFAULTS={"전진건설로봇":"079900.KS","페니트리움바이오":"187660.KQ","현대바이오사이언스":"048410.KQ","Moderna":"MRNA"}
if "watch" not in st.session_state: st.session_state.watch=DEFAULTS.copy()

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); u=d.clip(lower=0); dn=-d.clip(upper=0)
    au=u.ewm(alpha=1/n,adjust=False).mean(); ad=dn.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+au/ad.replace(0,np.nan))

def ind(d):
    x=d.copy()
    for n in (5,10,20,60,120,200): x[f"MA{n}"]=x.Close.rolling(n).mean()
    x["RSI"]=rsi(x.Close)
    x["MACD"]=ema(x.Close,12)-ema(x.Close,26); x["MS"]=ema(x.MACD,9); x["MH"]=x.MACD-x.MS
    x["V20"]=x.Volume.rolling(20).mean(); x["VR"]=x.Volume/x.V20
    pc=x.Close.shift(); tr=pd.concat([(x.High-x.Low),(x.High-pc).abs(),(x.Low-pc).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.rolling(14).mean()
    mid=x.Close.rolling(20).mean(); sd=x.Close.rolling(20).std()
    x["BBU"]=mid+2*sd; x["BBL"]=mid-2*sd
    x["HH20"]=x.High.shift(1).rolling(20).max()
    return x

@st.cache_data(ttl=120)
def daily(t):
    d=yf.download(t,period="3y",interval="1d",auto_adjust=False,progress=False,threads=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])

@st.cache_data(ttl=60)
def intra(t,interval):
    # Yahoo intraday history is limited; use a recent window only.
    period="5d" if interval in ("5m","15m","30m") else "1mo"
    d=yf.download(t,period=period,interval=interval,auto_adjust=False,progress=False,threads=False,prepost=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])

def tf(d,k):
    if k=="일": return ind(d)
    rule="W-FRI" if k=="주" else "ME"
    return ind(d.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna())

def basic(x):
    x=ind(x) if "RSI" not in x else x
    a=x.iloc[-1]; b=x.iloc[-2]; p=float(a.Close)
    def f(k,default=0): return float(a[k]) if k in a and pd.notna(a[k]) else default
    m20=f("MA20",p); m60=f("MA60",p); rv=f("RSI",50); vr=f("VR",0)
    sup=float(x.Low.tail(min(20,len(x))).min()); res=float(x.High.iloc[-21:-1].max()) if len(x)>21 else float(x.High.max())
    rebound=p>float(b.Close) and float(a.Close)>float(a.Open)
    trend="상승" if p>m20>m60 else ("하락" if p<m20<m60 else "혼조")
    score=50+(12 if trend=="상승" else -12 if trend=="하락" else 0)
    score+=8 if f("MACD")>f("MS") else -4
    score+=6 if 45<=rv<=68 else -5 if rv>=75 else 2 if rv<35 else 0
    score+=7 if rebound and vr>=1.2 else 0
    score=max(0,min(100,score))
    return dict(p=p,rsi=rv,vr=vr,sup=sup,res=res,trend=trend,score=score,macd="강세" if f("MACD")>f("MS") else "약세")

def intraday_analysis(d):
    if d is None or len(d)<10: return None
    x=d.copy()
    # Session VWAP resets each trading date.
    idx=pd.DatetimeIndex(x.index)
    datekey=idx.date
    tp=(x.High+x.Low+x.Close)/3
    pv=tp*x.Volume
    x["VWAP"]=pv.groupby(datekey).cumsum()/x.Volume.groupby(datekey).cumsum().replace(0,np.nan)
    x["EMA9"]=ema(x.Close,9); x["EMA20"]=ema(x.Close,20); x["RSI"]=rsi(x.Close)
    a=x.iloc[-1]; p=float(a.Close); vw=float(a.VWAP) if pd.notna(a.VWAP) else p
    recent=x.tail(min(20,len(x)))
    sup=float(recent.Low.min()); res=float(recent.High.iloc[:-1].max()) if len(recent)>1 else float(a.High)
    vbase=x.Volume.tail(min(21,len(x))).iloc[:-1].mean() if len(x)>1 else 0
    vr=float(a.Volume/vbase) if vbase else 0
    bullish=p>vw and p>float(a.EMA9)>float(a.EMA20)
    reclaim=(float(x.Close.iloc[-2])<=float(x.VWAP.iloc[-2]) and p>vw) if len(x)>2 and pd.notna(x.VWAP.iloc[-2]) else False
    breakout=p>res and vr>=1.3
    pullback=(abs(p-vw)/p<=.006 and p>=vw and float(a.Close)>float(a.Open))
    if breakout: sig="🟢 장중 돌파"
    elif reclaim and vr>=1.1: sig="🟢 VWAP 재돌파"
    elif pullback: sig="🔵 VWAP 눌림 확인"
    elif bullish: sig="🔵 단기 상승 유지"
    elif p<vw and p<float(a.EMA20): sig="🔴 단기 약세"
    else: sig="⚪ 대기"
    return dict(x=x,p=p,vwap=vw,rsi=float(a.RSI) if pd.notna(a.RSI) else 50,vr=vr,sup=sup,res=res,bull=bullish,sig=sig)

def money(v,t): return f"{v:,.0f}원" if t.endswith((".KS",".KQ")) else f"${v:,.2f}"

st.title("⚡ Stock Watch v5 · Intraday Layer")
st.caption("v4의 일·주·월 큰 추세 위에 5·15·30·60분봉 + 세션 VWAP을 얹은 장중 확인 버전")

with st.sidebar:
    st.header("관심종목")
    nm=st.text_input("종목 이름",placeholder="예: NVIDIA")
    tk=st.text_input("티커",placeholder="NVDA / 005930.KS / 247540.KQ").strip().upper()
    if st.button("➕ 추가",use_container_width=True) and tk:
        st.session_state.watch[nm.strip() or tk]=tk; st.rerun()
    if st.session_state.watch:
        dele=st.selectbox("삭제",["선택 안 함"]+list(st.session_state.watch))
        if st.button("➖ 삭제",use_container_width=True) and dele!="선택 안 함":
            st.session_state.watch.pop(dele,None); st.rerun()
    st.divider()
    st.caption("장중 데이터는 Yahoo 공급 상태에 따라 지연/누락될 수 있습니다. 주문용 실시간 API가 아닙니다.")

rows=[]; raws={}
for n,t in st.session_state.watch.items():
    try:
        d=daily(t); raws[t]=d; a=basic(tf(d,"일"))
        rows.append([n,t,a["score"],a["trend"],round(a["rsi"],1),round(a["vr"],2)])
    except Exception: rows.append([n,t,0,"조회 실패",np.nan,np.nan])

st.subheader("전체 감시판")
st.dataframe(pd.DataFrame(rows,columns=["종목","티커","일봉점수","일봉추세","RSI","거래량배수"]),hide_index=True,use_container_width=True)

if st.session_state.watch:
    name=st.selectbox("상세 분석 종목",list(st.session_state.watch))
    t=st.session_state.watch[name]
    d=raws.get(t,daily(t))
    frames={k:tf(d,k) for k in ("일","주","월")}
    big={k:basic(v) for k,v in frames.items() if len(v)>25}
    a=big["일"]

    st.subheader(f"{name} · {t}")
    c=st.columns(5)
    c[0].metric("최근 종가",money(a["p"],t)); c[1].metric("일봉점수",a["score"])
    c[2].metric("일봉 RSI",f'{a["rsi"]:.1f}'); c[3].metric("일봉 거래량",f'{a["vr"]:.2f}×'); c[4].metric("일봉추세",a["trend"])
    st.write("**큰 추세:** "+" · ".join(f"{k} {q['trend']}({q['score']})" for k,q in big.items()))
    st.write(f"**일봉 지지:** {money(a['sup'],t)} · **일봉 저항:** {money(a['res'],t)}")

    st.subheader("장중 타임프레임")
    tabs=st.tabs(["5분","15분","30분","60분"])
    intraday_results={}
    for tab,label,iv in zip(tabs,["5분","15분","30분","60분"],["5m","15m","30m","60m"]):
        with tab:
            try:
                q=intraday_analysis(intra(t,iv)); intraday_results[label]=q
                if q:
                    st.metric("장중 판정",q["sig"])
                    st.write(f"가격 **{money(q['p'],t)}** · VWAP **{money(q['vwap'],t)}** · RSI **{q['rsi']:.1f}** · 직전 평균대비 거래량 **{q['vr']:.2f}×**")
                    st.write(f"단기 지지 **{money(q['sup'],t)}** · 단기 저항 **{money(q['res'],t)}**")
                    st.line_chart(q["x"][["Close","VWAP","EMA9","EMA20"]].tail(100))
                else: st.info("장중 데이터 부족")
            except Exception as e: st.warning(f"{label} 데이터 조회 실패: {e}")

    good=sum(1 for q in intraday_results.values() if q and ("🟢" in q["sig"] or "상승 유지" in q["sig"]))
    bad=sum(1 for q in intraday_results.values() if q and "🔴" in q["sig"])
    st.subheader("장중 합의")
    if a["trend"]=="상승" and good>=2:
        st.success(f"🟢 큰 추세 상승 + 장중 {good}개 타임프레임 확인. 눌림/돌파 가격 확인 우선.")
    elif a["trend"]=="하락" and bad>=2:
        st.error(f"🔴 일봉 하락 + 장중 {bad}개 약세. 과매도만 보고 진입하지 않기.")
    else:
        st.info(f"⚪ 장중 합의 부족: 강세 {good}개 / 약세 {bad}개. 추가 확인 대기.")

    st.subheader("일·주·월 차트")
    btabs=st.tabs(["일봉","주봉","월봉"])
    for tab,k in zip(btabs,("일","주","월")):
        with tab:
            st.line_chart(frames[k][["Close","MA20","MA60"]].tail(150 if k=="일" else 80))

st.warning("아직 '진짜 실시간'은 아닙니다. 외국인·기관·프로그램·공매도, 뉴스/공시/임상 이벤트, 실시간 체결 알림과 주문은 증권사/시장 데이터 API 연결 단계에서 추가합니다.")
