import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Stock Watch v3", page_icon="📡", layout="wide")

DEFAULTS={"전진건설로봇":"079900.KS","페니트리움바이오":"187660.KQ","현대바이오사이언스":"048410.KQ","Moderna":"MRNA"}
if "watch" not in st.session_state: st.session_state.watch=DEFAULTS.copy()

def rsi(s,n=14):
    d=s.diff(); u=d.clip(lower=0); v=-d.clip(upper=0)
    au=u.ewm(alpha=1/n,adjust=False).mean(); av=v.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+au/av.replace(0,np.nan))

def ind(d):
    x=d.copy()
    for n in (5,20,60,120): x[f"MA{n}"]=x.Close.rolling(n).mean()
    x["RSI"]=rsi(x.Close)
    e12=x.Close.ewm(span=12,adjust=False).mean(); e26=x.Close.ewm(span=26,adjust=False).mean()
    x["MACD"]=e12-e26; x["MS"]=x.MACD.ewm(span=9,adjust=False).mean()
    x["V20"]=x.Volume.rolling(20).mean()
    return x

@st.cache_data(ttl=120)
def get(t):
    d=yf.download(t,period="2y",interval="1d",auto_adjust=False,progress=False,threads=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])

def tf(d,k):
    if k=="일": return ind(d)
    rule="W-FRI" if k=="주" else "ME"
    return ind(d.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna())

def judge(x):
    x=x.dropna(subset=["Close"]); a=x.iloc[-1]; b=x.iloc[-2]; p=float(a.Close)
    def val(k,default): return float(a[k]) if pd.notna(a[k]) else default
    m5,m20,m60,m120=[val(f"MA{n}",p) for n in (5,20,60,120)]
    rv=val("RSI",50); vr=float(a.Volume/a.V20) if pd.notna(a.V20) and a.V20 else 0
    sup=float(x.Low.tail(20).min()); res=float(x.High.iloc[-21:-1].max())
    rebound=p>float(b.Close) and float(a.Close)>float(a.Open)
    near=p<=sup*1.04 or m20*.98<=p<=m20*1.03 or m60*.98<=p<=m60*1.03
    s1=near and rebound and vr>=1.10
    old=x.High.shift(1).rolling(20).max()
    broke=bool((x.Close.iloc[-7:-1]>old.iloc[-7:-1]).any())
    s2=broke and res*.985<=p<=res*1.035 and p>=float(b.Close)
    drop=(x.Close.tail(8).min()/x.Close.tail(8).max()-1)<=-.07
    fade=x.Volume.tail(3).mean()<x.Volume.iloc[-8:-3].mean() if len(x)>=8 else False
    s3=drop and fade and rebound and vr>=1.10
    trend="상승" if p>m20>m60 else ("하락" if p<m20<m60 else "혼조")
    macd="강세" if a.MACD>a.MS else "약세"
    if s1 or s2 or s3:
        sig="🟢 추가매수 확인"; why="① 지지반등+거래량" if s1 else ("② 돌파 후 눌림지지" if s2 else "③ 급락 후 매도감소+반등거래")
    elif rv>=72 or (p>m20*1.10 and vr>=1.5):
        sig="🟠 추격 금지"; why="과열/이격 확대"
    elif p<sup*.97 or (p<m60 and a.MACD<a.MS):
        sig="🔴 위험"; why="지지 또는 중기추세 약화"
    else:
        sig="⚪ 대기"; why="3대 확인조건 미충족"
    return dict(p=p,r=rv,vr=vr,sup=sup,res=res,m5=m5,m20=m20,m60=m60,m120=m120,trend=trend,macd=macd,sig=sig,why=why,invalid=min(sup,m60)*.97)

def money(v,t): return f"{v:,.0f}원" if t.endswith((".KS",".KQ")) else f"${v:,.2f}"

st.title("📡 Stock Watch v3")
st.caption("다종목 전체 스캔 → 종목 선택 → 일·주·월 상세 분석")

with st.sidebar:
    st.header("관심종목")
    name=st.text_input("종목 이름",placeholder="예: NVIDIA")
    ticker=st.text_input("티커",placeholder="NVDA / 005930.KS / 247540.KQ").strip().upper()
    if st.button("➕ 추가",use_container_width=True) and ticker:
        st.session_state.watch[name.strip() or ticker]=ticker; st.rerun()
    if st.session_state.watch:
        delete=st.selectbox("삭제",["선택 안 함"]+list(st.session_state.watch))
        if st.button("➖ 삭제",use_container_width=True) and delete!="선택 안 함":
            st.session_state.watch.pop(delete,None); st.rerun()
    st.divider(); st.subheader("MRNA")
    shares=st.number_input("보유 주수",value=4.0,step=1.0)
    avg=st.number_input("평단($)",value=162.31,step=.01)
    limit=st.number_input("총 투자한도(원)",value=5000000,step=100000)

rows=[]; data={}
for n,t in st.session_state.watch.items():
    try:
        d=get(t); data[t]=d
        if len(d)<130: rows.append([n,t,"데이터 부족","-",np.nan,np.nan]); continue
        a=judge(tf(d,"일")); rows.append([n,t,a["sig"],a["trend"],round(a["r"],1),round(a["vr"],2)])
    except Exception: rows.append([n,t,"조회 실패","-",np.nan,np.nan])

st.subheader("전체 감시판")
st.dataframe(pd.DataFrame(rows,columns=["종목","티커","신호","추세","RSI","거래량배수"]),hide_index=True,use_container_width=True)

if st.session_state.watch:
    pick=st.selectbox("상세 분석",list(st.session_state.watch))
    t=st.session_state.watch[pick]
    try:
        d=data.get(t) if t in data else get(t)
        fs={k:tf(d,k) for k in ("일","주","월")}; a=judge(fs["일"])
        st.subheader(f"{pick} · {t}")
        c=st.columns(4); c[0].metric("현재/최근 종가",money(a["p"],t)); c[1].metric("RSI",f'{a["r"]:.1f}')
        c[2].metric("거래량/20일",f'{a["vr"]:.2f}×'); c[3].metric("판정",a["sig"])
        st.write(f"**근거:** {a['why']} | **지지:** {money(a['sup'],t)} | **저항:** {money(a['res'],t)} | **무효/위험:** {money(a['invalid'],t)} 하회")
        st.write(f"MA5 {money(a['m5'],t)} · MA20 {money(a['m20'],t)} · MA60 {money(a['m60'],t)} · MA120 {money(a['m120'],t)}")
        tabs=st.tabs(["일봉","주봉","월봉"])
        for tab,k in zip(tabs,("일","주","월")):
            with tab:
                x=fs[k]; st.line_chart(x[["Close","MA5","MA20","MA60","MA120"]].tail(150 if k=="일" else 80))
                if len(x)>22:
                    q=judge(x); st.caption(f"{k}봉 {q['trend']} · RSI {q['r']:.1f} · MACD {q['macd']} · 거래량 {q['vr']:.2f}×")
        if t=="MRNA":
            pnl=(a["p"]/avg-1)*100 if avg else 0
            st.info(f"{shares:g}주 · 평단 ${avg:.2f} · 평단 대비 {pnl:+.1f}% · 총 투자한도 약 {limit:,.0f}원")
    except Exception as e: st.error(f"상세 분석 실패: {e}")

st.caption("※ Yahoo Finance 기반 지연 데이터일 수 있습니다. 단순 하락/급등은 매수신호로 판정하지 않습니다. 실시간 체결·자동주문은 다음 단계에서 증권사 API로 연결합니다.")
