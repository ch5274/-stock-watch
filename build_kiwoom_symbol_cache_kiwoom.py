#!/usr/bin/env python3
import json, os, subprocess, tempfile, time
from pathlib import Path

ENV_PATH = Path.home() / ".kiwoom_env"
OUT = Path("/home/opc/kiwoom_symbols.json")
TOKEN_URL = "https://api.kiwoom.com/oauth2/token"
API_URL = "https://api.kiwoom.com/api/dostk/stkinfo"
API_ID = "ka10099"

def load_env():
    d = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d

def curl_json(url, headers, body, header_out=None):
    cmd = ["curl","-sS","-X","POST",url,
           "-H","Content-Type: application/json;charset=UTF-8"]
    for k,v in headers.items():
        cmd += ["-H", f"{k}: {v}"]
    if header_out:
        cmd += ["-D", header_out]
    cmd += ["-d", json.dumps(body, ensure_ascii=False)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "curl failed")
    try:
        return json.loads(p.stdout)
    except Exception:
        raise RuntimeError("JSON parse failed: " + p.stdout[:300])

def issue_token():
    e = load_env()
    data = curl_json(TOKEN_URL, {}, {
        "grant_type":"client_credentials",
        "appkey":e["KIWOOM_APP_KEY"],
        "secretkey":e["KIWOOM_APP_SECRET"]
    })
    if data.get("return_code") != 0 or not data.get("token"):
        raise RuntimeError(f"token failed: {data.get('return_code')} / {data.get('return_msg')}")
    return data["token"]

def read_cont_headers(path):
    cont_yn = ""
    next_key = ""
    for line in Path(path).read_text(errors="ignore").splitlines():
        low = line.lower()
        if low.startswith("cont-yn:"):
            cont_yn = line.split(":",1)[1].strip()
        elif low.startswith("next-key:"):
            next_key = line.split(":",1)[1].strip()
    return cont_yn, next_key

def fetch_market(token, market):
    rows = []
    cont_yn = ""
    next_key = ""
    for _ in range(40):
        with tempfile.NamedTemporaryFile(delete=False) as h:
            hp = h.name
        headers = {
            "authorization": f"Bearer {token}",
            "api-id": API_ID,
        }
        if cont_yn:
            headers["cont-yn"] = cont_yn
        if next_key:
            headers["next-key"] = next_key

        data = curl_json(API_URL, headers, {"mrkt_tp": market}, header_out=hp)
        if data.get("return_code") not in (None, 0):
            raise RuntimeError(f"ka10099 failed: {data.get('return_code')} / {data.get('return_msg')}")
        block = data.get("list", [])
        if isinstance(block, list):
            rows.extend(x for x in block if isinstance(x, dict))

        cont_yn, next_key = read_cont_headers(hp)
        try: os.unlink(hp)
        except Exception: pass

        if cont_yn != "Y":
            break
        time.sleep(0.25)
    return rows

def main():
    token = issue_token()
    all_rows = []
    for market in ("0","10"):  # KOSPI, KOSDAQ
        all_rows.extend(fetch_market(token, market))

    slim = []
    seen = set()
    for x in all_rows:
        code = str(x.get("code","")).replace("A","").strip()
        name = str(x.get("name","")).strip()
        if not code or not name or code in seen:
            continue
        seen.add(code)
        slim.append({
            "code": code,
            "name": name,
            "exchange": x.get("marketName",""),
            "marketCode": x.get("marketCode",""),
            "sector": x.get("upName",""),
            "state": x.get("state",""),
            "nxtEnable": x.get("nxtEnable","")
        })

    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUT)
    print(f"KIWOOM_SYMBOLS_OK {len(slim)}")

if __name__ == "__main__":
    main()
