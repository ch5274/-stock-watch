#!/usr/bin/env python3
import json, os
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

WATCH=Path("/home/opc/kiwoom_watchlist.json")
QUOTES=Path("/home/opc/kiwoom_realtime.json")
SYMBOLS=Path("/home/opc/kiwoom_symbols.json")
TOKEN=os.getenv("KIWOOM_BRIDGE_TOKEN","")
app=FastAPI(title="Stock Watch Kiwoom Bridge")

class Code(BaseModel):
    code:str

def auth(x_bridge_token):
    if TOKEN and x_bridge_token!=TOKEN:
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

@app.get("/search")
def search(q:str="",limit:int=20,x_bridge_token:str|None=Header(default=None)):
    auth(x_bridge_token)
    q=q.strip().lower()
    if not q: return {"results":[]}
    try: data=json.loads(SYMBOLS.read_text(encoding="utf-8"))
    except Exception: return {"results":[]}
    result=[]
    for x in data:
        name=str(x.get("name",""))
        code=str(x.get("code","")).zfill(6)
        if q in name.lower() or q in code:
            result.append({"name":name,"symbol":code,"code":code,
                           "exchange":x.get("exchange","KRX")})
            if len(result)>=min(max(limit,1),50): break
    return {"results":result}

@app.get("/quote/{code}")
def quote(code:str,x_bridge_token:str|None=Header(default=None)):
    auth(x_bridge_token)
    try: q=json.loads(QUOTES.read_text(encoding="utf-8"))
    except Exception: q={}
    return q.get(code.zfill(6),{})

@app.post("/watch/add")
def add(body:Code,x_bridge_token:str|None=Header(default=None)):
    auth(x_bridge_token)
    c=body.code.strip().replace("A","").zfill(6)
    x=load_watch()
    if c not in x: x.append(c); save_watch(x)
    return {"ok":True,"codes":x}

@app.post("/watch/remove")
def remove(body:Code,x_bridge_token:str|None=Header(default=None)):
    auth(x_bridge_token)
    c=body.code.strip().replace("A","").zfill(6)
    x=[v for v in load_watch() if v!=c]
    save_watch(x)
    return {"ok":True,"codes":x}

@app.get("/watch")
def watch(x_bridge_token:str|None=Header(default=None)):
    auth(x_bridge_token)
    return {"codes":load_watch()}
