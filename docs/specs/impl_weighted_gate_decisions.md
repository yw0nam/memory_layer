# 구현 스펙: 가중 합산 버스트 게이트 + 결정사항 추출 (Issue #9)

## 근거

Cerebras 원문: "each burst is scored against a **weighted combination of signals** and must
clear a threshold before it is embedded … reactions, providing a **social boost**."
→ strict AND(현행)를 원본 충실한 가중 합산으로 변경. 사용자 확정.

## 대상 파일

- 수정: `src/history_index.py`, `tests/test_selective_gates.py`(게이트 계약 변경분만)
- 신규 테스트: decision 추출 관련 (기존 파일에 추가 또는 신규 파일, 작성자 재량)
- 기타 src 파일 수정 금지. memory_chunks 스키마 변경 금지(컬럼 추가 불가 —
  chunk_kind 값으로만 구분, search.py 계약 유지).

## 변경 1 — `passes_burst_gate` 가중 합산으로

```python
def burst_score(mean_idf_value: float, has_social: bool) -> float:
    return mean_idf_value + (1.0 if has_social else 0.0)

def passes_burst_gate(burst: Burst, mean_idf_value: float, document_count: int) -> bool:
    # 원본: weighted combination clears threshold; social은 가점(+1.0)
    if document_count < 20:  # bootstrap: IDF 신뢰 불가 구간
        return True
    return burst_score(mean_idf_value, burst.social_weight > 1.0) >= 4.0
```

- 200자 필수는 기존 `group_bursts(min_chars=200)` 유지.
- 사회적 신호 정의 불변: 도구 에러 or 3분 내 사용자 재질문(`social_weight=1.5` 계산 로직 그대로).
- 부트스트랩(N<20)일 때는 social 무관 통과(길이 게이트만) — 기존 테스트
  `bootstrap_bypass_still_requires_social_signal`은 **새 계약에 맞게 수정**(아래 Step A).
- 통계 키 변경: `burst_no_social` → `burst_below_threshold` (IDF든 social이든 합산 미달).
  `low_idf_burst` 키는 제거하고 `burst_below_threshold`로 통합.

## 변경 2 — 세션 증류에 decisions 추가

- `_distill` 프롬프트의 JSON 스키마에 `"decisions"` 추가:
  `"decisions": ["세션에서 내려진 주요 결정사항. 무엇을 왜 그렇게 하기로 했는지 1문장씩. 없으면 빈 배열"]`
- `Distillation` 데이터클래스에 `decisions: list[str]` 필드 추가 (기본 `[]` 허용 파싱:
  누락/비리스트면 빈 배열, 항목은 str 변환·공백 제거·빈 항목 제거).
- `Distillation.text`(세션 행 임베딩 대상)는 **기존 4요소 유지** — decisions는 별도 행이므로
  세션 행에 중복 포함하지 않는다.
- 저장: 세션당 decision마다 행 추가 —
  - `id = f"{session_id}:decision:{i}"`, `chunk_kind = "decision"`
  - `content_raw = decision 원문`, `distilled = f"[{one_line_question}] 결정: {decision}"`
  - 임베딩은 distilled 대상, `idf_score = NULL`, `ts_last_active` = 세션과 동일
  - `metadata` = 세션 행과 동일 + `{"index": i}`
- decision 개수 상한 10개/세션 (LLM 폭주 방지, 초과분 버리고 경고 로그).

## Step A — 테스트 먼저 (red)

`tests/test_selective_gates.py` 게이트 부분을 새 계약으로 수정 + decision 파싱 테스트 추가:

- `burst_score`: 신호 유무 ±1.0 가점 검증.
- `passes_burst_gate` 새 계약: IDF 4.0 단독 통과 / IDF 3.0+신호(=4.0) 통과 /
  IDF 3.9 단독 탈락 / IDF 2.9+신호(=3.9) 탈락 / N<20이면 신호 없어도 통과(기존 테스트 반전) /
  N=20 경계는 부트스트랩 아님.
- decision 파싱(`Distillation` 생성 경로의 순수 부분): decisions 누락 → [], 비문자열 항목 정리,
  10개 초과 절단. 파싱 로직을 순수 함수로 분리 요구: `parse_distillation(parsed: dict) -> Distillation`
  (기존 `_distill` 내 파싱 부분을 추출한 것 — 구현 단계에서 `_distill`이 이를 호출하도록).
- red 확인: 새/수정 테스트가 현행 구현에서 실패해야 함 (기존 `burst_score`/`parse_distillation`
  부재 → ImportError, 게이트 동작 차이 → assert 실패).

## Step B — 구현 (green)

- 위 변경 1·2 구현. `parse_distillation` 순수 함수 분리 포함.
- 검증: `uv run pytest tests/test_selective_gates.py -q` green,
  `uv run pytest -m "not integration" -q` 전체 green.

## 수용 기준 (오케스트레이터 수행 포함)

1. Step A 직후 red, Step B 후 green. 기존 무관 테스트 전부 유지.
2. `--dry-run` 전체: burst 생존율이 strict AND(6개) 대비 증가하되 전량 통과 아님을 확인.
3. `--limit 10 --full` 실적재: decision 행 생성 확인, `search.py --source history`로 decision 히트 확인.
4. 증분 재실행 유지.
