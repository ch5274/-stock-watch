#!/usr/bin/env python3
import json, os
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

WATCH=Path("/home/opc/kiwoom_watchlist.json")
QUOTES=Path("/home/opc/kiwoom_realtime.json")
TICKS=Path("/home/opc/kiwoom_ticks.jsonl")
TOKEN=os.getenv("KIWOOM_BRIDGE_TOKEN","")
app=FastAPI(title="Stock Watch Kiwoom Bridge")

class Code(BaseModel):
    code:str

def auth(x_bridge_token):
    if TOKEN and x_bridge_token != TOKEN:
        raise HTTPException(401,"unauthorized")

def load_watch():
    try:
        x=json.loads(WATCH.read_text(encoding="utf-8"))
        return x if isinstance(x,list) else x.get("codes",[])
    except Exception: return []

def save_watch(codes):
    tmp=WATCH.with_suffix(".tmp")
    tmp.write_text(json.dumps(sorted(set(codes)),ensure_ascii=False),encoding="utf-8")
    tmp.replace(WATCH)

@app.get("/health")
def health(): return {"ok":True}

@app.get("/quote/{code}")
def quote(code:str, x_bridge_token:str|None=Header(default=None)):
    auth(x_bridge_token)
    try: q=json.loads(QUOTES.read_text(encoding="utf-8"))
    except Exception: q={}
    return q.get(code.zfill(6),{})

@app.get("/watch")
def watch(x_bridge_token:str|None=Header(default=None)):
    auth(x_bridge_token); return {"codes":load_watch()}

@app.post("/watch/add")
def add(body:Code, x_bridge_token:str|None=Header(default=None)):
    auth(x_bridge_token); c=body.code.strip().zfill(6); x=load_watch()
    if c not in x: x.append(c); save_watch(x)
    return {"ok":True,"message":f"{c} 감시 추가","codes":x}

@app.post("/watch/remove")
def remove(body:Code, x_bridge_token:str|None=Header(default=None)):
    auth(x_bridge_token); c=body.code.strip().zfill(6); x=[v for v in load_watch() if v!=c]
    save_watch(x); return {"ok":True,"message":f"{c} 감시 삭제","codes":x}


def _num(v):
    try: return abs(float(str(v).replace(",","").replace("+","")))
    except Exception: return None

def _bars(code, minutes, limit):
    import pandas as pd
    rows=[]
    if not TICKS.exists(): return []
    # Read tail only to keep the bridge light.
    with TICKS.open("rb") as f:
        try:
            f.seek(0,2); size=f.tell(); f.seek(max(0,size-4_000_000))
            if f.tell()>0: f.readline()
        except Exception: f.seek(0)
        for b in f:
            try:
                x=json.loads(b.decode("utf-8"))
                if x.get("code")==code:
                    p=_num(x.get("price")); v=_num(x.get("volume")) or 0
                    if p is not None: rows.append((x["ts"],p,v))
            except Exception: pass
    if not rows: return []
    df=pd.DataFrame(rows,columns=["ts","price","volume"])
    df["ts"]=pd.to_datetime(df["ts"])
    df=df.set_index("ts").sort_index()
    rule=f"{int(minutes)}min"
    o=df["price"].resample(rule).ohlc()
    vol=df["volume"].resample(rule).sum().rename("volume")
    z=o.join(vol).dropna().tail(limit).reset_index()
    return [{"ts":r.ts.isoformat(),"open":r.open,"high":r.high,"low":r.low,
             "close":r.close,"volume":r.volume} for r in z.itertuples()]

@app.get("/bars/{code}/{minutes}")
def bars(code:str, minutes:int, limit:int=200, x_bridge_token:str|None=Header(default=None)):
    auth(x_bridge_token)
    if minutes not in (1,5,15,30,60):
        raise HTTPException(400,"minutes must be 1,5,15,30,60")
    return {"code":code.zfill(6),"minutes":minutes,"bars":_bars(code.zfill(6),minutes,min(limit,500))}
