# 구현 스펙: Phase 2 — Planner→Executor→Synthesis + MCP 얇은 도구

## 목표

두 파일을 구현한다.

1. `src/answer.py` — 질의 → Planner(소스 선택) → Executor(하이브리드 검색) → Synthesis(인용 포함 답변) CLI.
2. `src/mcp_server.py` — 검색 기본기를 MCP stdio 서버의 얇은 도구로 노출: `search`, `search_code`, `search_history`.

## 컨텍스트 (기존 코드 — 수정 금지, 읽고 맞출 것)

- 프로젝트 루트: `/home/spow12/codes/2026_upper/agents/memory/memory_layer/repos/memory_base`
- Python 3.12, uv 프로젝트. 의존성: openai(AsyncOpenAI), asyncpg, httpx, numpy, python-dotenv 설치됨.
- `src/common.py`: `llm_client() -> AsyncOpenAI`(vLLM, base_url은 .env의 LLM_URL), `LLM_MODEL`.
- `src/search.py`: `async def search(query: str, source: str = "all", rerank: bool = True) -> list[Hit]`.
  `Hit` 필드: `source`("code"|"history"), `ref`(파일:라인 or 세션 ref), `text`, `ts`(epoch sec),
  `rrf`, `rerank_score`, `meta`(dict, code면 "context"에 인접 청크가 있을 수 있음).
  이미 FTS+벡터+시간감쇠+RRF+리랭커+맥락복원을 전부 수행한다. **검색 로직을 재구현하지 말 것.**
- 실행은 항상 저장소 루트에서 `uv run python src/answer.py ...` 형태. src/ 안의 모듈은
  서로 `from common import ...` 식 플랫 임포트를 쓴다(기존 관례 유지).

## 1) `src/answer.py`

```
uv run python src/answer.py "질문" [--source auto|code|history|all]
```

- **Planner**: `--source auto`(기본)일 때만 LLM 1회 호출. 질의를 보고
  `{"source": "code"|"history"|"all", "queries": ["검색어 1개~3개"]}` JSON 반환받는다
  (response_format json_object). 질의가 코드 구조 질문이면 code, "예전에/그때/어떻게 풀었지" 류
  회고 질문이면 history, 애매하면 all. queries는 원 질의를 검색 친화적으로 변형한 것
  (원 질의 자체도 반드시 포함).
- **Executor**: queries 각각에 대해 `search.search(q, source=결정된 소스)`를 **asyncio.gather로
  병렬** 실행 → Hit들을 (source, ref) 기준 중복제거, rerank_score(없으면 rrf) 내림차순 top-10.
- **Synthesis**: LLM 1회 호출. 시스템 프롬프트에 원칙 명시:
  "제공된 증거만 사용, 각 주장 뒤에 [1][2] 형태 인용, 증거가 부족하면 부족하다고 말할 것,
  오래된 정보(ts 오래됨)와 최신 정보가 충돌하면 최신 우선 + 주의사항 표기, 한국어로 답변."
  유저 메시지에 번호 붙인 증거 블록(각각 source/ref/날짜(ts를 YYYY-MM-DD로)/text, text는 2000자 절단,
  code Hit에 meta["context"] 있으면 함께) + 원 질문.
- 출력: 답변 본문 + 마지막에 "참조:" 섹션(번호 → ref 매핑).
- 증거 0개면 LLM 호출 없이 "관련 증거를 찾지 못했다"고 출력.

## 2) `src/mcp_server.py`

- 의존성: `uv add mcp` (공식 Python SDK). FastMCP 사용 (`from mcp.server.fastmcp import FastMCP`).
- stdio 서버, 서버명 "memory-base". **LLM 호출 없음** — 얇은 검색 도구만 (스펙 §3.2 ⑦).
- 도구 3개, 모두 `search.search()` 위임:
  - `search(query: str, top_k: int = 10)` → source="all"
  - `search_code(query: str, top_k: int = 10)` → source="code"
  - `search_history(query: str, top_k: int = 10)` → source="history"
- 반환: JSON 직렬화 가능한 list[dict] — `{"source", "ref", "date"(YYYY-MM-DD), "score"(rerank_score
  또는 rrf), "text"(2000자 절단), "context"(있으면)}`. top_k로 절단.
- docstring을 충실히: 도구 설명이 곧 에이전트가 보는 사용설명서다. 언제 어떤 도구를 쓸지 명시.
- 실행: `uv run python src/mcp_server.py` 로 stdio 서버가 뜬다.
- README 갱신 대신 파일 상단 docstring에 Claude Code 등록 예시 한 줄:
  `claude mcp add memory-base -- uv --directory <절대경로> run python src/mcp_server.py`

## 테스트 (필수 산출물)

`tests/test_answer_mcp.py` — **LLM·DB 없이** 실행 가능해야 함 (pytest, `uv add --dev pytest` 되어있지
않으면 추가):
- answer.py의 증거 중복제거·정렬·증거블록 포맷 함수를 순수 함수로 분리해 단위 검증.
- mcp_server의 Hit→dict 변환 함수를 순수 함수로 분리해 단위 검증 (Hit는 search.Hit 직접 생성).
- MCP 서버 in-process 검증: `mcp.shared.memory.create_connected_server_and_client_session` 또는
  FastMCP의 내장 테스트 유틸로 tools/list에 도구 3개가 뜨는지 확인. 이 부분이 SDK 버전상 어려우면
  도구 등록 객체 존재 확인으로 대체 가능(단, 이유를 테스트 주석에 남길 것).
- 검증에 monkeypatch로 `search.search`를 스텁해 네트워크/DB 접근 차단.

## 수용 기준

1. `uv run pytest tests/test_answer_mcp.py` 통과.
2. `uv run python src/answer.py "RRF 융합은 어디서 구현했지?" --source code` 가 인용 포함 한국어 답변 출력
   (실제 LLM/DB 필요 — 구현자 환경에서 동작 확인).
3. `uv run python src/mcp_server.py` 가 에러 없이 기동(stdio 대기).

## 금지 사항

- `src/common.py`, `src/search.py`, `src/code_index.py`, `src/history_index.py` 수정 금지.
- 검색 파이프라인(RRF/리랭크 등) 재구현 금지 — search.search() 호출만.
- mcp 외 새 의존성 추가 금지.
