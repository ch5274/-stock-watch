import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title='주식 감시기', page_icon='📡', layout='wide')

DEFAULTS = {
    '전진건설로봇':'079900.KS',
    'Moderna':'MRNA',
    '현대바이오사이언스':'048410.KQ',
}

@st.cache_data(ttl=300)
def load(ticker, period='1y', interval='1d'):
    d = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    return d.dropna()

def indicators(d):
    x=d.copy()
    for n in [5,20,60,120,200]: x[f'MA{n}']=x.Close.rolling(n).mean()
    delta=x.Close.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan); x['RSI']=100-(100/(1+rs))
    e12=x.Close.ewm(span=12,adjust=False).mean(); e26=x.Close.ewm(span=26,adjust=False).mean()
    x['MACD']=e12-e26; x['SIGNAL']=x.MACD.ewm(span=9,adjust=False).mean(); x['HIST']=x.MACD-x.SIGNAL
    x['VOL20']=x.Volume.rolling(20).mean()
    x['OBV']=(np.sign(x.Close.diff()).fillna(0)*x.Volume).cumsum()
    mid=x.Close.rolling(20).mean(); sd=x.Close.rolling(20).std(); x['BBU']=mid+2*sd; x['BBL']=mid-2*sd
    return x

def verdict(name,x):
    r=x.iloc[-1]; p=float(r.Close); vol=float(r.Volume); av=float(r.VOL20) if pd.notna(r.VOL20) else vol
    reasons=[]; score=0; signal='🟡 대기'
    if pd.notna(r.RSI):
        if 35<=r.RSI<=60: score+=1; reasons.append(f'RSI {r.RSI:.1f}: 과열 아님')
        elif r.RSI>=70: score-=1; reasons.append(f'RSI {r.RSI:.1f}: 과열')
    if pd.notna(r.MACD) and pd.notna(r.SIGNAL):
        if r.MACD>r.SIGNAL: score+=1; reasons.append('MACD 우위')
        else: reasons.append('MACD 약세')
    if av and vol>av*1.5: score+=1; reasons.append(f'거래량 20일평균의 {vol/av:.1f}배')
    if pd.notna(r.MA20) and p>r.MA20: score+=1; reasons.append('20일선 위')
    if name=='전진건설로봇':
        if 39000<=p<=40000: reasons.append('39~40천원 핵심 지지 시험')
        if p>44650 and vol>av*1.5: score+=2; reasons.append('44,650원 거래량 돌파')
        if p<38000 and vol>av*1.3: signal='🔴 위험'; reasons.append('38,000원 이탈 + 거래량 증가')
    if signal!='🔴 위험' and score>=4: signal='🟢 확인 필요'
    return signal,score,reasons

st.title('📡 멀티마켓 주식 감시기 v1')
st.caption('자동 시세 수집 → 기술지표 계산 → 조건 판정. 데이터 공급원 특성상 실시간이 아닌 지연 시세일 수 있습니다.')

with st.sidebar:
    st.header('관심종목')
    custom=st.text_input('티커 직접 입력', placeholder='예: 079900.KS, MRNA')
    names=list(DEFAULTS.keys())
    selected=st.multiselect('기본 관심종목', names, default=['전진건설로봇','Moderna'])
    if custom: selected.append(custom)

for item in selected:
    ticker=DEFAULTS.get(item,item); name=item
    try:
        d=indicators(load(ticker))
        if d.empty: st.warning(f'{name}: 데이터 없음'); continue
        r=d.iloc[-1]; signal,score,reasons=verdict(name,d)
        st.subheader(f'{name}  ·  {ticker}')
        c1,c2,c3,c4=st.columns(4)
        c1.metric('현재/최근 종가',f'{float(r.Close):,.2f}')
        c2.metric('RSI(14)',f'{float(r.RSI):.1f}' if pd.notna(r.RSI) else '-')
        c3.metric('거래량',f'{int(r.Volume):,}')
        c4.metric('판정',signal)
        st.line_chart(d[['Close','MA20','MA60']].tail(120))
        st.write('**판정 근거:** ' + ' · '.join(reasons))
        if name=='전진건설로봇':
            st.info('보유 기준: 32주 / 평단 66,750원. 감시: 39~40천 지지반등, 44,650 돌파·재지지, 48~50천 추세전환, 38천 이탈 위험.')
        with st.expander('상세 보조지표'):
            st.dataframe(d[['Close','Volume','MA5','MA20','MA60','MA120','MA200','RSI','MACD','SIGNAL','HIST','OBV','BBU','BBL']].tail(20), use_container_width=True)
    except Exception as e:
        st.error(f'{name}: {e}')
