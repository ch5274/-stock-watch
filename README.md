# Stock Watch V10.3

GitHub/Oracle용 통합본.

파일:
- app.py: V9 분석기 기반 V10.3 UI
- kiwoom_realtime_service.py: 키움 토큰 자동발급 + WebSocket 자동재접속 + 실시간 JSON 브리지
- requirements.txt: Python 의존성

보안:
- App Key / App Secret은 GitHub에 넣지 않습니다.
- Oracle의 ~/.kiwoom_env를 그대로 사용합니다.

현재 단계:
- 국내주식 현재가: 키움 실시간 브리지 우선
- 일/주/월 및 분봉 기술지표: 기존 Yahoo 이력 데이터 유지
- 다음 단계: 키움 체결 누적으로 5/15/30/60분봉 완전 실시간화
