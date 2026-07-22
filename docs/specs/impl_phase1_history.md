# 구현 스펙: Phase 1 — 에이전트 이력 수집·증류 파이프라인

## 목표

Claude Code 세션 JSONL(`~/.claude/projects/<encoded-path>/<session-id>.jsonl`)을 파싱해
LLM으로 증류(distill)하고, `memory.memory_chunks` 테이블에 임베딩과 함께 적재하는
**증분·멱등** 파이프라인 `src/history_index.py`를 구현한다.

## 컨텍스트 (기존 코드 — 수정 금지, 읽고 맞출 것)

- 프로젝트 루트: `/home/spow12/codes/2026_upper/agents/memory/memory_layer/repos/memory_base`
- Python 3.12, uv 프로젝트. 의존성 설치됨: asyncpg, openai(AsyncOpenAI), numpy, python-dotenv, httpx, cocoindex(무관)
- `src/common.py`: `VllmEmbedder().embed(text, query=False) -> np.float16[2048]` (async),
  `llm_client() -> AsyncOpenAI`, `LLM_MODEL`, `DB_URL`, `PG_SCHEMA = "memory"`. 반드시 재사용.
- DB: Postgres(포트 5439, docker `memory_base_db`), pgvector 0.8.5, 스키마 `memory` 존재.
- `memory.code_chunks` 테이블은 CocoIndex가 관리 — **절대 건드리지 말 것**.
- `src/search.py`의 `_search_history()`가 이미 아래 컬럼을 SELECT한다(계약):
  `id, source_ref, distilled, content_raw, ts_last_active, idf_score, embedding(halfvec)`.
  이 계약을 깨지 말 것. 실행: `cd src && uv run python search.py "질의" --source history`

## 테이블 (이 파이프라인이 소유·생성, IF NOT EXISTS)

```sql
CREATE TABLE IF NOT EXISTS memory.memory_chunks (
  id             text PRIMARY KEY,          -- "{session_id}:session" 또는 "{session_id}:burst:{n}"
  source_type    text NOT NULL,             -- 'claude_code' (Hermes 등은 추후 어댑터)
  source_ref     text NOT NULL,             -- "{project_dir_name}/{session_id}" (사람이 추적 가능)
  chunk_kind     text NOT NULL,             -- 'session' | 'burst'
  session_id     text NOT NULL,
  content_raw    text NOT NULL,             -- FTS 대상 원문(세션이면 재구성 트랜스크립트, 버스트면 버스트 원문)
  distilled      text,                      -- 임베딩된 텍스트(세션: 증류 결과, 버스트: 주제 프리픽스+버스트)
  embedding      halfvec(2048) NOT NULL,
  ts_last_active double precision NOT NULL, -- 세션 마지막 메시지 epoch sec
  idf_score      double precision,          -- 버스트 mean-IDF (세션 행은 NULL 허용)
  metadata       jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS memory_chunks__fts ON memory.memory_chunks
  USING GIN (to_tsvector('simple', content_raw));
CREATE INDEX IF NOT EXISTS memory_chunks__vec ON memory.memory_chunks
  USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS memory_chunks__session ON memory.memory_chunks (session_id);
-- 증분 상태 추적
CREATE TABLE IF NOT EXISTS memory.ingest_state (
  file_path text PRIMARY KEY,
  mtime     double precision NOT NULL,
  size      bigint NOT NULL,
  ingested_at double precision NOT NULL
);
-- 코퍼스 토큰 DF (IDF 계산용)
CREATE TABLE IF NOT EXISTS memory.df_stats (
  token text PRIMARY KEY,
  df    bigint NOT NULL
);
-- 총 문서 수는 df_stats에 특수 토큰 '__N__'으로 저장하거나 별도 1행 테이블. 구현 재량.
```

## JSONL 파싱 규칙 (실물 포맷 확인됨)

한 줄 = JSON 객체. `type` 필드 기준:

- **취급 대상**: `type in ("user", "assistant")` 만. 그 외(`attachment, ai-title, mode,
  system, file-history-snapshot, queue-operation, bridge-session, ...`)는 전부 skip.
- `isSidechain == true` 인 줄은 skip (서브에이전트 트래픽).
- 공통 필드: `timestamp`(ISO8601, "2026-06-25T05:08:12.347Z"), `sessionId`, `cwd`, `gitBranch`, `uuid`, `parentUuid`.
- **user**: `message.content`가 str이면 그대로 텍스트. list이면 블록 배열 —
  `{"type":"text"}` 블록의 text만 취하고, `{"type":"tool_result"}` 블록은 텍스트로 넣지 말되
  `is_error` 필드를 도구 실패 신호로 집계한다.
- **assistant**: `message.content`는 블록 list — `{"type":"text"}`의 text만 이어붙인다.
  `{"type":"thinking"}` skip. `{"type":"tool_use"}`는 텍스트로 넣지 말되 도구명(name)을 집계.
- 빈 텍스트 메시지는 버린다.
- 파일이 손상 줄(파싱 불가 JSON)을 포함할 수 있음 — 해당 줄만 skip.

## 세션 재구성 → 행 생성

**세션 = 스레드.** 세션당:

1. **session 행 1개**: `content_raw` = "USER: ...\nASSISTANT: ..." 형태 재구성 트랜스크립트
   (100k자 초과 시 head 60% + "\n...[truncated]...\n" + tail 40%로 절단).
   `distilled` = LLM 증류 결과(아래), 임베딩은 distilled에 대해 수행.
2. **burst 행 0~N개**: 동일 화자 연속 텍스트 메시지 묶음(버스트) 중
   **결합 길이 ≥ 200자 AND mean-IDF ≥ 4.0** 인 것만.
   `distilled` = `"[{세션 주제 한 줄}] {버스트 원문}"` (주제는 증류 결과의 one_line_question 재사용),
   임베딩은 이 distilled에 대해 수행. `content_raw` = 버스트 원문.
   - **사회적 가중(스펙 §3.2)**: 버스트 구간에 도구 에러(is_error) 또는 직후 사용자 재질문
     (같은 사용자가 3분 내 재발화)이 있으면 `metadata.social_weight = 1.5`, 아니면 1.0.
     idf_score에 곱하지 말고 metadata에만 기록(검색측에서 추후 활용).

메시지 5개 미만이거나 총 텍스트 500자 미만인 세션은 skip (노이즈).

## LLM 증류

- `common.llm_client()` + `LLM_MODEL` 사용, JSON 모드(response_format={"type":"json_object"}).
- 프롬프트에 트랜스크립트(절단본)를 넣고 다음 JSON 추출:
  `{"one_line_question": "...", "summary": "...", "resolution": "...", "references": ["파일/시스템/명령 언급"]}`
  - one_line_question: "나중에 이 세션을 찾을 때 던질 법한 검색 질문 한 줄"
  - summary: 3~5문장 요약, resolution: 최종 해결책/결론(없으면 "미해결")
  - 출력 언어: 한국어(원문에 등장하는 코드·에러문자열·식별자는 원문 유지).
- session 행의 `distilled` = one_line_question + "\n" + summary + "\n" + resolution + "\n" + ", ".join(references)
- LLM 호출 실패(타임아웃/거부) 시: 그 세션은 distilled=None으로 두지 말고 **skip하고 경고 로그**
  (ingest_state에 기록하지 않아 다음 실행에서 재시도되게).

## IDF

- 토큰화: `re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}|[가-힣]{2,}", text.lower())` 수준의 단순 토크나이저.
- DF 갱신: 세션 단위 문서로 카운트(세션의 고유 토큰 집합 → df += 1). N = 총 세션 수.
- `idf(t) = ln((N+1)/(df(t)+1)) + 1`. 버스트 mean-IDF = 버스트 고유 토큰 idf 평균.
- 부트스트랩 문제(초기 N 작음)는 무시 — N < 20 이면 IDF 필터를 통과시킨다(길이 조건만 적용).

## 증분·멱등

- 대상 파일: `~/.claude/projects/*/*.jsonl` (약 145개). glob 후 `ingest_state`와 mtime+size 비교,
  변한 파일만 처리. `--full`이면 전부 재처리.
- **활성 세션 보호**: mtime이 최근 10분 이내인 파일은 skip(아직 기록 중).
- 세션 재처리 시: 트랜잭션 안에서 `DELETE FROM memory_chunks WHERE session_id=$1` 후 재삽입.
- 임베딩 호출은 세션당 순차로 충분(PoC). LLM 증류는 동시 4개까지 세마포어.

## CLI

```
uv run python src/history_index.py [--limit N] [--project SUBSTR] [--full] [--dry-run]
```
- `--limit N`: 최신 mtime 순 N개 파일만 (테스트용)
- `--project SUBSTR`: 프로젝트 디렉토리명 부분일치 필터
- `--dry-run`: DB 쓰기 없이 파싱·필터 통계만 출력
- 실행 종료 시 요약 출력: 처리 파일 수, 생성 세션/버스트 행 수, skip 사유별 카운트.

## 테스트 (필수 산출물)

`tests/test_history_parse.py` — DB·LLM·임베딩 **없이** 실행 가능해야 함:
- 인라인 픽스처(위 포맷의 JSONL 문자열 몇 줄)로 파서 단위 검증:
  sidechain skip, thinking skip, tool_result 텍스트 미포함 + is_error 집계, 트랜스크립트 재구성.
- 버스트 그룹핑과 200자 필터 검증.
- 실행: `uv run pytest tests/test_history_parse.py` (pytest는 `uv add --dev pytest`로 추가).
- 이를 위해 파싱·버스팅 로직은 I/O 없는 순수 함수로 분리할 것.

## 수용 기준

1. `uv run pytest tests/test_history_parse.py` 통과.
2. `uv run python src/history_index.py --limit 5` 성공, memory_chunks에 행 생성.
3. `cd src && uv run python search.py "아무 관련 질의" --source history` 가 결과 반환(에러 없이).
4. 같은 명령 재실행 시 변경 없는 파일은 재처리하지 않음(증분 동작, 로그로 확인 가능).

## 금지 사항

- `src/common.py`, `src/search.py`, `src/code_index.py` 수정 금지.
- 새 무거운 의존성 추가 금지(pytest dev 의존성만 허용). 표준 라이브러리 + 기존 설치분으로 구현.
- memory.code_chunks 테이블 접근 금지.
