#!/usr/bin/env python3
import json
from pathlib import Path
from pykrx import stock

OUT=Path("/home/opc/kiwoom_symbols.json")
rows=[]
for market in ("KOSPI","KOSDAQ","KONEX"):
    for code in stock.get_market_ticker_list(market=market):
        rows.append({"code":code,"name":stock.get_market_ticker_name(code),"exchange":market})
tmp=OUT.with_suffix(".tmp")
tmp.write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8")
tmp.replace(OUT)
print(f"saved {len(rows)} symbols")
