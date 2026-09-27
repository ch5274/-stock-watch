# Stock Watch V10.5 Realtime Charts

V10.4 기능을 모두 유지하면서 국내주식 실시간 차트 엔진을 추가한 버전.

추가 기능
- 키움 0B 실시간 체결 누적
- 1분 원천 데이터에서 5/15/30/60분 OHLCV 생성
- 실시간 캔들 차트
- 실시간 RSI(14), MACD, VWAP
- 관심종목 추가/삭제와 Oracle 감시목록 연동 유지
- 연결 오류 시 새 토큰 발급 + 자동 재접속 유지

보안
- App Key / App Secret은 GitHub에 포함하지 않음
- ~/.kiwoom_env 사용

다음 확장
- 실시간 MFI/OBV/ADX/ATR 및 기술점수 완전 실시간화
- 수급/공시/뉴스/재무/밸류/촉매/리스크 통합점수 엔진
