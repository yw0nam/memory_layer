# 구현 스펙: 테스트 커버리지 확충 (Issue #5)

## 목표

기존 구현(수동 검증만 된 상태)에 대한 **특성화·통합 테스트**를 작성해 회귀 방지 기반을 만든다.
이 작업 자체는 테스트만 작성한다 — **src/ 코드 수정 금지** (버그를 발견하면 수정하지 말고
테스트에 `@pytest.mark.xfail(reason=...)` 을 달고 보고서에 기록).

## 컨텍스트

- 저장소: memory_base (개인 RAG·에이전트 메모리 시스템). 작업은 지정된 워크트리 디렉토리에서 수행.
- 기존 테스트: `tests/test_answer_mcp.py`(12개), `tests/test_history_parse.py`(3개) — 유지되어야 함.
- 대상 모듈(전부 `src/` 아래, 플랫 임포트 관례):
  - `search.py`: `_rrf_fuse(lists)`, `_apply_time_decay(hits)`, `_dedup_cap(hits)`, `Hit` 데이터클래스,
    `search(query, source, rerank)` (DB+임베더+리랭커 필요)
  - `history_index.py`: `mean_idf`, `tokenize`, `build_transcript`, `group_sessions` (순수 함수 추가 커버)
  - `mcp_server.py`: FastMCP 서버 `mcp`, 도구 3개
  - `answer.py`: `plan`/`execute`/`synthesize` (LLM 필요)
- 실행 환경: `.env`에 LLM_URL/EMB_URL/RERANK_URL/DB_URL 있음. DB는 docker `memory_base_db`(포트 5439),
  이미 code_chunks 34행·memory_chunks 38행 적재 상태.
- pytest 설정: `pyproject.toml`에 `[tool.pytest.ini_options]`로 `integration` 마커 등록 필요
  (이 파일 수정은 허용 — 마커 등록·testpaths에 한함).

## 산출물

### 1. `tests/test_search_unit.py` — DB/네트워크 무의존

- `_rrf_fuse`: k=60 공식 검증 — 두 리스트 모두 1위인 문서 점수 = 2/(60+1); 한 리스트에만 있는
  문서보다 두 리스트 합의 문서가 높은 점수(합의 > 단일 강한 표) 검증.
- `_apply_time_decay`: 같은 rrf의 두 Hit에서 90일 오래된 쪽이 정확히 절반(반감기 90일) 검증,
  age 0은 감쇠 없음.
- `_dedup_cap`: 같은 파일 청크 5개 → 3개로 상한(PER_FILE_CAP), 전체 FUSED_TOP=20 상한,
  rrf 내림차순 유지 검증.
- `Hit` 기본값(meta dict 독립성 등) 간단 검증.

### 2. `tests/test_history_unit.py` — DB/LLM 무의존

- `mean_idf`: N/df 수식 검증(알려진 소형 코퍼스로 손계산 값 비교), 빈 텍스트 → 0.0.
- `tokenize`: 식별자(`snake_case`, CamelCase 소문자화)와 한글 2자+ 추출, 1글자·기호 제외.
- `build_transcript`: 100k 초과 시 head 60%+marker+tail 40% 구조 검증.
- `group_sessions`: 세션별 그룹핑, ts_last_active = 메시지 최대 timestamp, fallback 동작.

### 3. `tests/test_integration.py` — `@pytest.mark.integration`, 실 서비스 사용

모듈 상단에서 DB 접속 실패 시 `pytest.skip(allow_module_level=True)`로 전체 스킵(CI 안전).

- `search(q, source="code", rerank=False)` → Hit 1개 이상, source=="code", ref에 ":L" 포함,
  rrf > 0 검증. (rerank=False로 리랭커 의존 없이 빠르게)
- `search(q, source="history", rerank=False)` → source=="history" Hit 반환 검증.
- `search(q, source="all", rerank=True)` → 리랭커 경로 포함 전체 파이프라인, rerank_score 존재 검증.
- FTS 정확 토큰 검증: code_chunks에 확실히 존재하는 리터럴(예: "halfvec")로 검색 시 해당 파일 히트.
- MCP in-process: `mcp.shared.memory.create_connected_server_and_client_session` 등으로
  `search_code` 도구를 실제 호출해 JSON 결과 스키마(source/ref/date/score/text) 검증.
  (SDK상 어려우면 `mcp_server._run_search` 직접 호출로 대체하고 주석에 이유 명시)
- LLM 통합(answer.plan)은 **1개만**: 코드 질문 → source가 "code" 또는 "all"이고 queries에 원 질의
  포함 검증. (LLM 비결정성 감안해 느슨한 단언만)

### 4. `pyproject.toml` 마커 등록

```toml
[tool.pytest.ini_options]
markers = ["integration: requires local DB/vLLM services"]
testpaths = ["tests"]
```

## 수용 기준

1. `uv run pytest -m "not integration" -q` → 기존 15개 + 신규 단위 전부 통과, 네트워크/DB 접근 없음.
2. `uv run pytest -m integration -q` → 통과 (이 환경에서 실 서비스로 실행).
3. `uv run pytest -q` → 전체 통과.
4. src/ 파일 diff 없음 (`git status`로 확인, pyproject.toml 마커 추가만 허용).

## 금지 사항

- src/ 코드 수정 금지 (버그 발견 시 xfail + 보고).
- 새 의존성 추가 금지 (pytest-asyncio 없음 — async는 asyncio.run() 패턴, 기존 테스트 참고).
- 통합 테스트에서 DB에 행을 쓰지 말 것(읽기 전용; history_index 실행 금지).
