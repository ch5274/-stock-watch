# Stock Watch V9.2 · Kiwoom 중심 기준본

V9 화면과 기존 분석엔진을 유지하면서 종목검색을 Yahoo에서 분리했습니다.

- 종목명 일부/6자리 코드 검색: Oracle symbol cache
- 실제 현재가/실시간 데이터: 기존 Kiwoom 실시간 엔진
- 관심종목 추가/삭제: Kiwoom bridge watchlist
- App Key/Secret은 GitHub에 포함하지 않음

Oracle 최초 1회:
python3 build_kiwoom_symbol_cache.py

주의:
종목명/코드 목록은 KRX 상장 마스터를 캐시한 것이고, 가격·실시간 시세의 데이터 소스는 Kiwoom입니다.
