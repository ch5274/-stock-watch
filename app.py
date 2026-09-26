import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Stock Watch v4", page_icon="📡", layout="wide")
DEFAULTS={"전진건설로봇":"079900.KS","페니트리움바이오":"187660.KQ","현대바이오사이언스":"048410.KQ","Moderna":"MRNA"}
if "watch" not in st.session_state: st.session_state.watch=DEFAULTS.copy()

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); u=d.clip(lower=0); dn=-d.clip(upper=0)
    au=u.ewm(alpha=1/n,adjust=False).mean(); ad=dn.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+au/ad.replace(0,np.nan))

def indicators(d):
    x=d.copy()
    for n in (5,10,20,60,120,200): x[f"MA{n}"]=x.Close.rolling(n).mean()
    x["RSI"]=rsi(x.Close)
    lo=x.RSI.rolling(14).min(); hi=x.RSI.rolling(14).max()
    x["STOCH_RSI"]=100*(x.RSI-lo)/(hi-lo).replace(0,np.nan)
    x["MACD"]=ema(x.Close,12)-ema(x.Close,26); x["MACD_SIG"]=ema(x.MACD,9); x["MACD_H"]=x.MACD-x.MACD_SIG
    x["VOL20"]=x.Volume.rolling(20).mean(); x["VOL_RATIO"]=x.Volume/x.VOL20
    mid=x.Close.rolling(20).mean(); sd=x.Close.rolling(20).std()
    x["BB_MID"]=mid; x["BB_UP"]=mid+2*sd; x["BB_LO"]=mid-2*sd; x["BB_WIDTH"]=(x.BB_UP-x.BB_LO)/mid
    pc=x.Close.shift(1); tr=pd.concat([(x.High-x.Low),(x.High-pc).abs(),(x.Low-pc).abs()],axis=1).max(axis=1)
    x["ATR"]=tr.rolling(14).mean(); x["ATR_PCT"]=100*x.ATR/x.Close
    up=x.High.diff(); dn=-x.Low.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    atr=tr.ewm(alpha=1/14,adjust=False).mean()
    pdi=100*pd.Series(plus,index=x.index).ewm(alpha=1/14,adjust=False).mean()/atr
    mdi=100*pd.Series(minus,index=x.index).ewm(alpha=1/14,adjust=False).mean()/atr
    x["PDI"]=pdi; x["MDI"]=mdi; x["ADX"]=(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)).ewm(alpha=1/14,adjust=False).mean()
    tp=(x.High+x.Low+x.Close)/3; mt=tp.diff()
    pmf=(tp*x.Volume).where(mt>0,0).rolling(14).sum(); nmf=(tp*x.Volume).where(mt<0,0).rolling(14).sum()
    x["MFI"]=100-100/(1+pmf/nmf.replace(0,np.nan))
    x["OBV"]=(np.sign(x.Close.diff()).fillna(0)*x.Volume).cumsum()
    md=(tp-tp.rolling(20).mean()).abs().rolling(20).mean()
    x["CCI"]=(tp-tp.rolling(20).mean())/(0.015*md.replace(0,np.nan))
    hh=x.High.rolling(14).max(); ll=x.Low.rolling(14).min()
    x["WILLR"]=-100*(hh-x.Close)/(hh-ll).replace(0,np.nan)
    x["ROC"]=100*x.Close.pct_change(12)
    x["HH20"]=x.High.shift(1).rolling(20).max(); x["LL20"]=x.Low.shift(1).rolling(20).min()
    return x

@st.cache_data(ttl=120)
def get_data(t):
    d=yf.download(t,period="3y",interval="1d",auto_adjust=False,progress=False,threads=False)
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    return d.dropna(subset=["Close"])

def timeframe(d,k):
    if k=="일": return indicators(d)
    rule="W-FRI" if k=="주" else "ME"
    q=d.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    return indicators(q)

def div_signal(x,n=20):
    z=x.tail(n)
    if len(z)<n or z.RSI.isna().all(): return "없음"
    half=n//2
    p1=z.Close.iloc[:half].min(); p2=z.Close.iloc[half:].min()
    r1=z.RSI.iloc[:half].min(); r2=z.RSI.iloc[half:].min()
    if p2<p1 and r2>r1: return "상승 다이버전스 후보"
    p1h=z.Close.iloc[:half].max(); p2h=z.Close.iloc[half:].max()
    r1h=z.RSI.iloc[:half].max(); r2h=z.RSI.iloc[half:].max()
    if p2h>p1h and r2h<r1h: return "하락 다이버전스 후보"
    return "없음"

def analyze(x):
    x=x.dropna(subset=["Close"]); a=x.iloc[-1]; b=x.iloc[-2]; p=float(a.Close)
    def f(k,default=np.nan): return float(a[k]) if k in a and pd.notna(a[k]) else default
    mas={n:f(f"MA{n}",p) for n in (5,10,20,60,120,200)}
    rv=f("RSI",50); vr=f("VOL_RATIO",0); atrp=f("ATR_PCT",0); adx=f("ADX",0)
    sup=float(x.Low.tail(20).min()); res=float(x.High.iloc[-21:-1].max())
    rebound=p>float(b.Close) and float(a.Close)>float(a.Open)
    near=(p<=sup*1.04) or any(m*.98<=p<=m*1.03 for m in (mas[20],mas[60]))
    s1=near and rebound and vr>=1.10
    breakout_recent=bool((x.Close.iloc[-7:-1]>x.HH20.iloc[-7:-1]).fillna(False).any())
    s2=breakout_recent and res*.985<=p<=res*1.035 and p>=float(b.Close)
    drop=(x.Close.tail(8).min()/x.Close.tail(8).max()-1)<=-.07
    fade=x.Volume.tail(3).mean()<x.Volume.iloc[-8:-3].mean() if len(x)>=8 else False
    s3=drop and fade and rebound and vr>=1.10
    s4=(p>mas[20] and float(b.Close)<=float(x.MA20.iloc[-2])*1.02 and rebound and vr>=1.05)
    tight=(x.High.tail(10).max()/x.Low.tail(10).min()-1)<.07
    s5=tight and p>res and vr>=1.4
    spread=max(mas[5],mas[10],mas[20])-min(mas[5],mas[10],mas[20])
    compressed=spread/p<.025
    s6=compressed and p>mas[5]>mas[20] and vr>=1.15
    s7=p>res and vr>=1.3
    pattern=[i for i,v in enumerate((s1,s2,s3,s4,s5,s6,s7),1) if v]

    score=50
    score += 12 if p>mas[20]>mas[60] else (-12 if p<mas[20]<mas[60] else 0)
    score += 7 if mas[5]>mas[10]>mas[20] else 0
    score += 7 if f("MACD")>f("MACD_SIG") and f("MACD_H")>float(x.MACD_H.iloc[-2]) else -3
    score += 5 if 45<=rv<=68 else (-6 if rv>=75 else 2 if rv<35 else 0)
    score += 6 if f("PDI")>f("MDI") and adx>=20 else 0
    score += 5 if f("MFI",50)>50 else -2
    score += 5 if x.OBV.iloc[-1]>x.OBV.rolling(20).mean().iloc[-1] else -2
    score += 7 if vr>=1.2 and rebound else (-5 if vr>=1.5 and not rebound else 0)
    score += 10 if pattern else 0
    score=max(0,min(100,score))

    trend="상승" if p>mas[20]>mas[60] else ("하락" if p<mas[20]<mas[60] else "혼조")
    if pattern: verdict="🟢 추가매수 확인"
    elif rv>=72 or p>mas[20]*1.10: verdict="🟠 추격 금지"
    elif p<sup*.97 or (p<mas[60] and f("MACD")<f("MACD_SIG")): verdict="🔴 손상/위험"
    elif score>=65: verdict="🔵 돌파 확인 대기"
    else: verdict="⚪ 관망"

    h60=float(x.High.tail(60).max()); l60=float(x.Low.tail(60).min()); rng=h60-l60
    fib={"38.2%":h60-.382*rng,"50%":h60-.5*rng,"61.8%":h60-.618*rng}
    bbpos=(p-f("BB_LO",p))/(f("BB_UP",p)-f("BB_LO",p)) if f("BB_UP",p)!=f("BB_LO",p) else .5
    return dict(p=p,score=score,verdict=verdict,trend=trend,pattern=pattern,rsi=rv,vr=vr,
                macd="강세" if f("MACD")>f("MACD_SIG") else "약세",adx=adx,mfi=f("MFI",50),
                stoch=f("STOCH_RSI",50),cci=f("CCI",0),willr=f("WILLR",-50),roc=f("ROC",0),
                atrp=atrp,bbpos=bbpos,sup=sup,res=res,invalid=min(sup,mas[60])*.97,
                mas=mas,div=div_signal(x),fib=fib)

def money(v,t): return f"{v:,.0f}원" if t.endswith((".KS",".KQ")) else f"${v:,.2f}"
PAT={1:"지지 반등 + 거래량",2:"돌파 후 눌림 지지",3:"급락 후 매도감소 + 반등거래",4:"20일선 눌림 재상승",5:"박스권 거래량 돌파",6:"이평 수렴→확산",7:"전고점 거래량 돌파"}

st.title("📡 Stock Watch v4 · Confluence Engine")
st.caption("큰 추세 → 가격구조 → 보조지표 합의 → 진입패턴 → 리스크 순서로 판정")

with st.sidebar:
    st.header("관심종목 관리")
    nm=st.text_input("종목 이름",placeholder="예: NVIDIA")
    tk=st.text_input("티커",placeholder="NVDA / 005930.KS / 247540.KQ").strip().upper()
    if st.button("➕ 추가",use_container_width=True) and tk:
        st.session_state.watch[nm.strip() or tk]=tk; st.rerun()
    if st.session_state.watch:
        delete=st.selectbox("삭제할 종목",["선택 안 함"]+list(st.session_state.watch))
        if st.button("➖ 삭제",use_container_width=True) and delete!="선택 안 함":
            st.session_state.watch.pop(delete,None); st.rerun()
    st.divider(); st.subheader("MRNA 포지션")
    shares=st.number_input("보유 주수",value=4.0,step=1.0)
    avg=st.number_input("평단($)",value=162.31,step=.01)
    limit=st.number_input("총 투자한도(원)",value=5000000,step=100000)

rows=[]; raw={}
for name,t in st.session_state.watch.items():
    try:
        d=get_data(t); raw[t]=d
        a=analyze(timeframe(d,"일"))
        rows.append([name,t,a["verdict"],a["score"],a["trend"],round(a["rsi"],1),round(a["vr"],2)])
    except Exception:
        rows.append([name,t,"조회 실패",0,"-",np.nan,np.nan])

st.subheader("전체 감시판")
st.dataframe(pd.DataFrame(rows,columns=["종목","티커","판정","점수","추세","RSI","거래량배수"]),
             hide_index=True,use_container_width=True)

if st.session_state.watch:
    pick=st.selectbox("상세 분석 종목",list(st.session_state.watch))
    t=st.session_state.watch[pick]
    try:
        d=raw.get(t,get_data(t)); fs={k:timeframe(d,k) for k in ("일","주","월")}
        A={k:analyze(v) for k,v in fs.items() if len(v)>25}
        a=A["일"]
        st.subheader(f"{pick} · {t}")
        c=st.columns(5)
        c[0].metric("현재/최근 종가",money(a["p"],t)); c[1].metric("종합점수",f'{a["score"]}/100')
        c[2].metric("RSI",f'{a["rsi"]:.1f}'); c[3].metric("거래량",f'{a["vr"]:.2f}×'); c[4].metric("판정",a["verdict"])

        st.write("**멀티 타임프레임:** "+" · ".join(f"{k} {v['trend']}({v['score']})" for k,v in A.items()))
        st.write(f"**지지:** {money(a['sup'],t)} · **저항:** {money(a['res'],t)} · **무효/위험:** {money(a['invalid'],t)} 하회")
        if a["pattern"]: st.success("감지 패턴: "+" / ".join(PAT[i] for i in a["pattern"]))
        else: st.info("현재 7개 진입패턴의 완성 신호 없음")

        tab1,tab2,tab3,tab4=st.tabs(["추세·차트","모멘텀","변동성·수급대용","가격구조"])
        with tab1:
            st.line_chart(fs["일"][["Close","MA5","MA10","MA20","MA60","MA120","MA200"]].tail(220))
            st.write(f"MACD **{a['macd']}** · ADX **{a['adx']:.1f}** · RSI 다이버전스 **{a['div']}**")
        with tab2:
            st.write(f"RSI **{a['rsi']:.1f}** · Stoch RSI **{a['stoch']:.1f}** · CCI **{a['cci']:.1f}** · Williams %R **{a['willr']:.1f}** · ROC **{a['roc']:.1f}%**")
        with tab3:
            st.write(f"ATR **{a['atrp']:.2f}%** · Bollinger 위치 **{a['bbpos']*100:.0f}%** · MFI **{a['mfi']:.1f}** · 거래량 **{a['vr']:.2f}×**")
            st.caption("OBV/MFI는 가격·거래량 기반 자금흐름 대용지표입니다. 외국인·기관·프로그램·공매도 실제 수급은 별도 데이터 API가 필요합니다.")
        with tab4:
            st.write("**60일 스윙 피보나치:** "+ " · ".join(f"{k} {money(v,t)}" for k,v in a["fib"].items()))
            st.write("**패턴 체크:** "+ " / ".join(f"{i}. {PAT[i]} {'✅' if i in a['pattern'] else '—'}" for i in PAT))

        st.subheader("일·주·월")
        tabs=st.tabs(["일봉","주봉","월봉"])
        for tab,k in zip(tabs,("일","주","월")):
            with tab:
                x=fs[k]
                st.line_chart(x[["Close","MA20","MA60"]].tail(150 if k=="일" else 80))
                if k in A:
                    q=A[k]; st.caption(f"{k}봉 {q['trend']} · 점수 {q['score']} · RSI {q['rsi']:.1f} · MACD {q['macd']} · 거래량 {q['vr']:.2f}×")

        if t=="MRNA":
            pnl=(a["p"]/avg-1)*100 if avg else 0
            st.info(f"MRNA {shares:g}주 · 평단 ${avg:.2f} · 평단 대비 {pnl:+.1f}% · 총 투자한도 약 {limit:,.0f}원. 임상/규제 이벤트는 기술점수와 별도 확인.")

        st.warning("현재 미구현 데이터: 외국인·기관·프로그램 수급, 공매도, 뉴스/공시·임상 일정, 섹터 상대강도, 진짜 장중 VWAP/5·15·30·60분봉. 다음 단계에서 실시간/시장 데이터 API로 연결합니다.")
    except Exception as e:
        st.error(f"상세 분석 실패: {e}")

st.caption("※ Yahoo Finance 기반 지연 데이터일 수 있습니다. 지표 개수보다 '추세+가격구조+거래량+패턴'의 합의를 우선하며, 단순 과매도나 급락만으로 매수 판정하지 않습니다.")
