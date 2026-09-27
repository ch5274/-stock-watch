#!/usr/bin/env python3
import asyncio, json, os, random, subprocess
from datetime import datetime
from pathlib import Path

SOCKET_URL="wss://api.kiwoom.com:10000/api/websocket"
ENV_PATH=os.path.expanduser("~/.kiwoom_env")
WATCH_FILE=Path("/home/opc/kiwoom_watchlist.json")
OUT_FILE=Path("/home/opc/kiwoom_realtime.json")
DEFAULT_CODES=["039490"]

def log(s): print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {s}",flush=True)

def load_env():
    d={}
    with open(ENV_PATH,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line and not line.startswith("#") and "=" in line:
                k,v=line.split("=",1); d[k.strip()]=v.strip()
    return d

def watch_codes():
    try:
        x=json.loads(WATCH_FILE.read_text(encoding="utf-8"))
        codes=x if isinstance(x,list) else x.get("codes",[])
        codes=[str(c).zfill(6) for c in codes if str(c).strip()]
        return codes or DEFAULT_CODES
    except Exception:
        return DEFAULT_CODES

def issue_token():
    e=load_env()
    payload=json.dumps({"grant_type":"client_credentials","appkey":e["KIWOOM_APP_KEY"],
                        "secretkey":e["KIWOOM_APP_SECRET"]})
    p=subprocess.run(["curl","-sS","-X","POST","https://api.kiwoom.com/oauth2/token",
        "-H","Content-Type: application/json;charset=UTF-8","-d",payload],
        capture_output=True,text=True,timeout=30)
    data=json.loads(p.stdout)
    if p.returncode or data.get("return_code")!=0 or not data.get("token"):
        raise RuntimeError(f"토큰 실패: {data.get('return_code')} / {data.get('return_msg')}")
    log("토큰 자동 발급 성공")
    return data["token"]

def atomic_write(obj):
    tmp=OUT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False),encoding="utf-8")
    tmp.replace(OUT_FILE)

def extract_realtime(msg,state):
    # Kiwoom REAL messages can contain a list of realtime records. Preserve raw fields,
    # and map common 0B stock-execution FIDs when present.
    rows=msg.get("data")
    if not isinstance(rows,list): return
    now=datetime.now().isoformat(timespec="seconds")
    for row in rows:
        if not isinstance(row,dict): continue
        code=str(row.get("item") or row.get("stk_cd") or row.get("code") or "").replace("A","")
        vals=row.get("values") if isinstance(row.get("values"),dict) else row
        if not code: continue
        rec=state.get(code,{})
        # Common Kiwoom stock execution field IDs; raw values are also retained.
        price=vals.get("10") or vals.get("cur_prc") or vals.get("price")
        volume=vals.get("15") or vals.get("trde_qty") or vals.get("volume")
        if price is not None: rec["price"]=price
        if volume is not None: rec["volume"]=volume
        rec["updated_at"]=now
        rec["raw"]=vals
        state[code]=rec
    atomic_write(state)

async def session():
    import websockets
    token=issue_token(); codes=watch_codes(); state={}
    async with websockets.connect(SOCKET_URL,ping_interval=None,close_timeout=5,max_size=None) as ws:
        log("WebSocket 연결 성공")
        await ws.send(json.dumps({"trnm":"LOGIN","token":token}))
        while True:
            raw=await asyncio.wait_for(ws.recv(),timeout=90)
            m=json.loads(raw); tr=m.get("trnm")
            if tr=="PING": await ws.send(raw); continue
            if tr=="LOGIN":
                if m.get("return_code")!=0: raise RuntimeError("LOGIN 실패")
                log("LOGIN 성공")
                await ws.send(json.dumps({"trnm":"REG","grp_no":"1","refresh":"1",
                    "data":[{"item":codes,"type":["0B"]}]}))
                log("실시간 등록: "+",".join(codes)); continue
            if tr=="REG":
                log(f"REG: {m.get('return_code')} / {m.get('return_msg')}"); continue
            extract_realtime(m,state)

async def main():
    wait=3
    while True:
        try: await session(); wait=3
        except asyncio.CancelledError: raise
        except Exception as e:
            log(f"오류: {type(e).__name__}: {e}")
            await asyncio.sleep(wait+random.random()); wait=min(60,wait*2)

if __name__=="__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: log("서비스 종료")
