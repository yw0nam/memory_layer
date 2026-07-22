# 구현 스펙: 선별 게이트 강화 + 2패스 적재 (Issue #7)

## 원칙 (Cerebras 원본)

**"저신호 데이터는 DB에 도달하지 않는다."** 게이트를 통과한 것만 증류(LLM 비용)·임베딩·저장한다.

## 대상 파일

- 수정: `src/history_index.py` (이 파일만)
- 테스트: `tests/test_selective_gates.py` (신규)
- 기존 46개 테스트는 계속 통과해야 함. 단, 기존 `tests/test_history_unit.py`의 `mean_idf` 등
  순수 함수 계약은 유지되므로 충돌 없을 것.

## 변경 1 — 2패스 적재: 코퍼스 DF를 매 실행 전체 재계산

현재: df_stats/history_session_tokens 테이블로 증분 DF 회계 (복잡, N<20 부트스트랩 무력).

변경:
- **Pass 1 (통계 패스)**: 매 실행 시작 시, 선택된 전체 파일(변경 여부 무관)을 파싱·토큰화해
  in-memory로 DF/N 재계산. LLM·임베딩·DB 쓰기 없음. 개인 규모(145파일)에서 수 초.
  - 새 순수 함수: `build_corpus_df(transcripts: Iterable[str]) -> tuple[dict[str, int], int]`
    — 각 transcript의 고유 토큰 집합으로 df 카운트, N = transcript 수.
  - 유효 세션(`_valid_session` 통과)의 transcript만 코퍼스에 포함.
- **Pass 2 (적재 패스)**: 기존처럼 변경 파일만 처리하되, IDF 계산은 Pass 1의 in-memory DF 사용.
- **제거**: `df_stats`, `history_session_tokens` 테이블과 그 회계 로직(`_corpus_without_file`의
  DF 부분, `_write_file`의 DF 갱신) 전부 삭제. `_ensure_schema`에서
  `DROP TABLE IF EXISTS ...df_stats`, `...history_session_tokens` 실행(파생 캐시라 안전).
  `history_file_sessions`는 멱등 재적재(세션 삭제)용으로 **유지**.
- `mean_idf` 함수 시그니처·수식은 유지(기존 테스트 보호).
- N<20 부트스트랩 우회는 유지하되, 이제 전체 코퍼스 기준이라 실질적으로는 비활성.

## 변경 2 — 세션 트리아지 게이트 (증류 전 저비용 선별)

새 순수 함수 `triage_heuristic(session: Session) -> str` — "keep" | "skip" | "borderline":

판정 순서(위에서부터 첫 매칭):
1. assistant 텍스트 메시지가 0개 → "skip"
2. user 텍스트 메시지 ≤ 2개 AND user 텍스트 합계 < 200자 → "skip"  (단순 명령 세션)
3. 전체 텍스트 합계 ≥ 5000자 OR 메시지 수 ≥ 20 → "keep"
4. 그 외 → "borderline"

- "borderline"만 LLM 저비용 판정: `_triage_llm(session) -> bool(keep)`.
  프롬프트: 트랜스크립트 앞 3000자를 주고 "나중에 '이 문제 어떻게 풀었지?'로 검색할 가치가 있는
  세션인가"를 `{"keep": true|false, "reason": "..."}` JSON으로. temperature=0.
  LLM 실패 시 **keep으로 fail-open**(데이터 손실 방지) + 경고 로그.
- `--dry-run`에서는 LLM 호출 없이 휴리스틱만 적용, borderline은 별도 카운트.
- 기존 `_valid_session`(메시지≥5, 500자)은 최소 게이트로 유지하고 그 다음에 트리아지 적용.
- skip된 세션도 멱등성을 위해 파일 처리 자체는 완료로 기록(ingest_state) — 단, 행은 저장 안 함.

## 변경 3 — 버스트 게이트에 사회적 신호 편입

새 순수 함수 `passes_burst_gate(burst: Burst, mean_idf_value: float, document_count: int) -> bool`:

```
길이 게이트: group_bursts의 min_chars=200 유지 (기존)
통과 = (document_count < 20 or mean_idf_value >= 4.0) and burst.social_weight > 1.0
```

- social_weight > 1.0 = 버스트에 도구 에러가 있거나 3분 내 사용자 재질문이 따라온 경우(기존 계산 유지).
- Cerebras의 "리액션 포함" AND 게이트의 에이전트 이력 대응(스펙 §3.2). 통과율이 지나치게 낮으면
  나중에 튜닝할 수 있도록 게이트 함수를 한 곳에 모아둘 것.
  `# ponytail: strict AND gate, relax to scoring if recall suffers` 주석 명시.

## 변경 4 — 게이트 통계 가시화

summary 출력에 추가 (Counter 키):
- `triage_keep`, `triage_skip_heuristic`, `triage_borderline`, `triage_llm_keep`, `triage_llm_skip`
- `burst_no_social` (IDF는 통과했지만 social로 탈락), `low_idf_burst` (기존 유지)
- dry-run에서도 동일 통계 출력 (LLM 판정 자리는 borderline 카운트로).

## TDD 절차 (중요)

이 작업은 두 단계로 나뉜다:

**Step A (테스트 작성자)**: `tests/test_selective_gates.py`에 위 인터페이스 대상 실패 테스트를 먼저 작성.
- `build_corpus_df`: 소형 코퍼스 손계산 DF/N 검증.
- `triage_heuristic`: 4개 규칙 각각 + 우선순위(assistant 0개가 규칙 3보다 우선) 검증.
- `passes_burst_gate`: idf/social 4조합 + N<20 우회 검증.
- import는 `from history_index import build_corpus_df, triage_heuristic, passes_burst_gate`.
  아직 함수가 없으므로 **ImportError로 실패(red)하는 것이 정상**.
- DB/LLM/네트워크 무의존.

**Step B (구현자)**: history_index.py를 수정해 테스트를 green으로. 스펙 외 동작 변경 금지.

## 수용 기준

1. `uv run pytest tests/test_selective_gates.py -q` — Step A 직후 red(ImportError), Step B 후 green.
2. `uv run pytest -m "not integration" -q` — 기존 포함 전부 통과.
3. `uv run python src/history_index.py --dry-run` (전체 파일) — 게이트 통계 출력, LLM/DB 접근 없음.
4. (오케스트레이터 수행) `--limit 10` 실적재로 선별 동작·스킵 통계 확인, 재실행 증분 유지.

## 금지 사항

- history_index.py 외 src 파일 수정 금지. 새 의존성 금지.
- memory_chunks 스키마 변경 금지 (search.py 계약 유지).
