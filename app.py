import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="주식 감시기 v2", page_icon="📡", layout="wide")
STOCKS={"전진건설로봇":"079900.KS","페니트리움바이오":"187660.KQ","현대바이오사이언스":"048410.KQ","Moderna":"MRNA"}

def RSI(s,n=14):
    d=s.diff(); u=d.clip(lower=0); v=-d.clip(upper=0)
    au=u.ewm(alpha=1/n,adjust=False).mean(); av=v.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+au/av.replace(0,np.nan))

def ind(d):
    x=d.copy()
    for n in [5,20,60,120]: x[f"MA{n}"]=x.Close.rolling(n).mean()
    x["RSI"]=RSI(x.Close); e12=x.Close.ewm(span=12,adjust=False).mean(); e26=x.Close.ewm(span=26,adjust=False).mean()
    x["MACD"]=e12-e26; x["MS"]=x.MACD.ewm(span=9,adjust=False).mean(); x["V20"]=x.Volume.rolling(20).mean()
    return x

@st.cache_data(ttl=300)
def load(t):
    d=yf.download(t,period="2y",interval="1d",auto_adjust=False,progress=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])

def resample(d,kind):
    if kind=="일": return ind(d)
    rule="W-FRI" if kind=="주" else "ME"
    return ind(d.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna())

def assess(x):
    a=x.iloc[-1]; b=x.iloc[-2]; p=float(a.Close)
    ma20=float(a.MA20) if pd.notna(a.MA20) else p; ma60=float(a.MA60) if pd.notna(a.MA60) else p
    r=float(a.RSI) if pd.notna(a.RSI) else 50; vr=float(a.Volume/a.V20) if pd.notna(a.V20) and a.V20 else 0
    sup=float(x.Low.tail(20).min()); resistance=float(x.High.iloc[-21:-1].max())
    rebound=p>float(b.Close) and float(a.Close)>float(a.Open)
    s1=(p<=sup*1.04 or ma20*.98<=p<=ma20*1.03) and rebound and vr>=1.10
    old_high=x.High.shift(1).rolling(20).max()
    breakout=bool((x.Close.iloc[-6:-1]>old_high.iloc[-6:-1]).any())
    s2=breakout and resistance*.985<=p<=resistance*1.035 and p>=float(b.Close)
    drop=(x.Close.tail(8).min()/x.Close.tail(8).max()-1)<=-.07
    fade=x.Volume.tail(3).mean()<x.Volume.iloc[-8:-3].mean()
    s3=drop and fade and rebound and vr>=1.10
    trend="상승" if p>ma20>ma60 else ("하락" if p<ma20<ma60 else "혼조")
    mc="강세" if a.MACD>a.MS else "약세"
    if s1 or s2 or s3: sig="🟢 추가매수 확인"; why="지지반등+거래량" if s1 else ("돌파후 눌림지지" if s2 else "급락후 매도감소+반등거래")
    elif r>=72 or (p>ma20*1.10 and vr>=1.5): sig="🟠 추격 금지"; why="과열/이격 확대"
    elif p<sup*.97 or (p<ma60 and a.MACD<a.MS): sig="🔴 위험"; why="지지 또는 중기추세 약화"
    else: sig="⚪ 대기"; why="우선 매수조건 미충족"
    return dict(p=p,r=r,vr=vr,sup=sup,res=resistance,ma20=ma20,ma60=ma60,trend=trend,mc=mc,sig=sig,why=why,invalid=min(sup,ma60)*.97)

def fmt(v,t): return f"{v:,.0f}원" if t.endswith((".KS",".KQ")) else f"${v:,.2f}"

st.title("📡 멀티마켓 주식 감시기 v2")
st.caption("일·주·월 + MA · RSI · MACD · 거래량 · 지지/저항 · 확인형 추가매수 신호")

with st.sidebar:
    chosen=st.multiselect("관심종목",list(STOCKS),default=list(STOCKS))
    custom=st.text_input("추가 티커",placeholder="NVDA / 005930.KS / 247540.KQ").strip().upper()
    st.divider(); st.subheader("MRNA 보유정보")
    shares=st.number_input("보유 주수",value=4.0,step=1.0); avg=st.number_input("평단($)",value=162.31,step=.01)
    limit=st.number_input("총 투자한도(원)",value=5000000,step=100000)

items=[(n,STOCKS[n]) for n in chosen]
if custom: items.append((custom,custom))
summary=[]
for name,t in items:
    try:
        d=load(t)
        if len(d)<130: st.warning(f"{name}: 데이터 부족"); continue
        frames={k:resample(d,k) for k in ["일","주","월"]}
        a=assess(frames["일"])
        with st.container(border=True):
            st.subheader(f"{name} · {t}")
            c=st.columns(4); c[0].metric("현재/최근 종가",fmt(a["p"],t)); c[1].metric("RSI",f'{a["r"]:.1f}')
            c[2].metric("거래량/20일",f'{a["vr"]:.2f}×'); c[3].metric("신호",a["sig"])
            st.write(f"**근거:** {a['why']} | **지지:** {fmt(a['sup'],t)} | **저항:** {fmt(a['res'],t)} | **무효/위험:** {fmt(a['invalid'],t)} 하회")
            tabs=st.tabs(["일봉","주봉","월봉"])
            for tab,k in zip(tabs,["일","주","월"]):
                with tab:
                    x=frames[k]
                    st.line_chart(x[["Close","MA20","MA60"]].tail(120 if k=="일" else 60))
                    if len(x)>22:
                        q=assess(x); st.caption(f"{k}봉 {q['trend']} · RSI {q['r']:.1f} · MACD {q['mc']} · 거래량 {q['vr']:.2f}×")
            if t=="MRNA":
                pnl=(a["p"]/avg-1)*100 if avg else 0
                st.info(f"보유 {shares:g}주 · 평단 ${avg:.2f} · 평단 대비 {pnl:+.1f}% · 총 투자한도 약 {limit:,.0f}원. 임상/이벤트 직전 급등 추격은 별도 확인.")
        summary.append([name,a["sig"],a["trend"],round(a["r"],1),round(a["vr"],2)])
    except Exception as e: st.error(f"{name} 조회 실패: {e}")

if summary:
    st.subheader("한눈에 보기")
    st.dataframe(pd.DataFrame(summary,columns=["종목","신호","일봉추세","RSI","거래량배수"]),hide_index=True,use_container_width=True)
st.caption("※ Yahoo Finance 기반 지연 시세일 수 있습니다. 단순 하락·급등은 매수신호로 보지 않습니다. 뉴스/임상 촉매는 이 버전에서 자동 판정하지 않습니다.")
