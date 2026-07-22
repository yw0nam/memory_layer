# 개인용 RAG · 에이전트 메모리 시스템 — 아키텍처 개요 + 로드맵

> 기반 소스
> - Cerebras Knowledge Base 설계 (원문 블로그 / PyTorchKR 해설) — pgvector · 하이브리드 검색 · LLM 증류 · 버스팅 · RRF · MCP
> - PIKE-RAG (arXiv 2501.11551, Microsoft Research Asia) — 특화 지식·근거(rationale) 증강, 다층 지식베이스, 지식 원자화, L0~L4 단계 로드맵
> - 스코프: **개인 / PoC 규모**, 우선순위 **① 코드 저장소 · ② 에이전트 채팅 이력(커스텀 DB) → ③ n8n · Gmail · GDrive · Slack · 위키**
>
> 이 문서의 기술적 수치·모델명은 2026년 7월 기준 웹 검증을 거쳤습니다(하단 "검증된 사실" 참조).

---

## 1. 한 줄 요약

Cerebras가 사내 15,000 질문/일을 처리하려고 만든 RAG 아키텍처를 **1인 개발자 규모로 압축**하고, 가장 값진 개인 데이터인 **코드 저장소 + 에이전트(Claude Code·Hermes 등) 대화 이력**을 하나의 pgvector 테이블에 모아 자연어로 질의하는 **개인 지식·메모리 시스템**. 고도화 축은 PIKE-RAG의 다층 지식베이스·지식 원자화·단계별(L0~L4) 능력 확장을 얹는 것.

핵심 설계 철학 3가지 (Cerebras에서 그대로 계승):
1. **데이터가 있는 곳으로 간다** — 소스를 옮기지 않고 각 소스에서 직접 추출한다.
2. **원본을 그대로 임베딩하지 않는다** — LLM으로 구조화 증류한 뒤 임베딩한다.
3. **단일 모델이 아니라 신호 융합** — 벡터 검색은 만능이 아니며 전문검색·IDF·시간감쇠와 RRF로 합쳐야 실용 정확도가 나온다.

---

## 2. 왜 이 조합인가 (개인 스코프에 맞춘 재해석)

| Cerebras 원본 문제 | 개인 스코프에서의 의미 |
|---|---|
| Slack이 가장 중요·가장 지저분한 소스 | **에이전트 채팅 이력**이 그 자리를 차지. 잡담 한 줄과 긴 디버깅 로그가 한 세션에 섞여 있음 → 동일한 하이브리드 문제 |
| 40GB+ 코드 저장소 증분 유지 | 로컬 저장소 몇 개. 규모는 작지만 **증분 인덱싱**은 여전히 핵심(매 커밋마다 전체 재계산 불가) |
| 사내 위키·Docs·Jira | 후순위 소스(Gmail·GDrive·n8n·Slack·위키) — 커넥터로 나중에 추가 |
| `who_knows`(전문가 검색) | 개인 스코프에선 불필요. 대신 **"내가 예전에 이 문제 어떻게 풀었지?"**(에피소드 메모리)가 그 자리 |

**메모리 시스템 관점**: 에이전트 대화 이력은 표준 메모리 분류(semantic/episodic/procedural/working)에서 **에피소드 + 절차 메모리**에 해당한다. Claude Code는 기본 30일 후 트랜스크립트를 삭제하므로(`cleanupPeriodDays`), 이 시스템에 증류·적재하면 **30일 창을 넘는 영구 장기기억**이 된다 — 이것이 "메모리 시스템"이라는 이름값의 핵심 동기다.

---

## 3. 아키텍처 개요

### 3.1 전체 계층 (Cerebras 6계층을 개인용으로 축약)

```
소스 → 수집 → 증류 → [단일 임베딩 테이블] → 하이브리드 검색 → 융합·재순위 → 종합
                                                                    ↑
                                                        MCP 얇은 도구로도 노출
```

```mermaid
flowchart TD
    subgraph SRC["① 소스 (우선순위순)"]
      A1[코드 저장소<br/>git repos]
      A2[에이전트 이력<br/>Claude Code JSONL · Hermes · SQLite]
      A3[후순위: n8n · Gmail · GDrive · Slack · 위키]
    end
    subgraph ING["② 수집 (커넥터 / 플러그인 스크립트)"]
      B1[CocoIndex<br/>Tree-sitter 증분]
      B2[이력 파서<br/>세션→스레드 재구성]
      B3[커스텀 커넥터]
    end
    subgraph DIST["③ 증류 (LLM 구조화)"]
      C1[요약 · 한줄질문 · 해결책 · 참조 추출]
      C2[버스팅: 세션 내 개별 신호 살리기]
      C3[PIKE-RAG: 지식 원자화 · 오토태깅]
    end
    DB[(④ 단일 Postgres + pgvector<br/>임베딩 · 원본요약 · 메타데이터<br/>+ FTS(GIN) 인덱스)]
    subgraph RET["⑤ 하이브리드 검색"]
      D1[전문검색 FTS]
      D2[임베딩 검색]
      D3[IDF 가중]
      D4[시간 감쇠]
    end
    subgraph FUSE["⑥ 융합 · 재순위"]
      E1[RRF k=60]
      E2[소스 중복제거 · 상한]
      E3[리랭커 top-10]
      E4[맥락 복원]
    end
    F[⑦ Planner → Executor → Synthesis]
    MCP{{MCP 얇은 도구<br/>search · search_code · search_history}}

    A1-->B1; A2-->B2; A3-->B3
    B1-->C1; B2-->C1; B3-->C1
    C1-->C2-->C3-->DB
    DB-->D1 & D2 & D3 & D4
    D1 & D2 & D3 & D4-->E1-->E2-->E3-->E4-->F
    DB-.->MCP
    MCP-.->F
```

### 3.2 컴포넌트별 설계

**① 소스 & ② 수집**

- **코드 저장소 (우선 1)**: **CocoIndex** 채택. 2026년 현재 Tree-sitter(구문 정확) + 언어별 정규식 경계로 청킹하고, **변경된 청크만 재임베딩**하는 증분 엔진. 동기화 메타데이터를 Postgres에 추적하므로 임베딩 저장소와 같은 DB에 둘 수 있음. 파일 수준·함수 수준 등 **다중 구체성** 임베딩 생성.
- **에이전트 이력 (우선 1, 커스텀 DB)**: 자체 **플러그인 스크립트**로 처리(Cerebras의 커스텀 소스 방식 그대로). 
  - Claude Code: `~/.claude/projects/<encoded-path>/<session-id>.jsonl` — 각 줄이 `{type: user|assistant|tool, content, timestamp, sessionId, cwd}`. 전역 인덱스 `~/.claude/history.jsonl`, 메타 `sessions-index.json`(요약·메시지 수·git 브랜치).
  - Hermes / 기타: JSON 또는 SQLite. 어댑터 하나가 소스별로 "우리 임베딩 테이블 형태의 행"을 내보내면 나머지 스택은 무변경.
  - **세션 = Slack 스레드**로 취급 → 세션 전체를 하나의 대화 상태로 재구성해 한 행으로 저장(Cerebras 스레드 재구성과 동형).
- **후순위 소스**: n8n(워크플로 로그/노트), Gmail, GDrive, Slack, 위키 → 각각 커넥터로 추가. 모두 **동일 스키마 행**을 공유 DB에 쓰면 끝(플러그인 아키텍처).

**③ 증류 (LLM 구조화)** — 정확도의 가장 큰 지렛대

- 원본을 그대로 임베딩하지 않는다. 스레드/세션마다 LLM이 다음을 추출: **검색될 법한 한 줄 질문 · 짧은 요약 · 해결책(resolution) · 언급된 시스템·코드 참조**.
- **버스팅**: 긴 세션에서 요약에 안 담긴 개별 신호를 살리기 위해, 동일 화자 연속 메시지 묶음(버스트)을 스레드 주제를 앞에 붙여 개별 임베딩. 저신호 차단 임계값 — IDF ≥ 4.0, 결합 200자 이상, (Slack이면 리액션 가중). **에이전트 이력에선 "리액션" 대신 도구 성공/에러, 사용자 재질문 여부 등을 사회적 가중치로 대체.**
- 원본 텍스트는 도착 즉시 FTS(GIN) 인덱스로 키워드 검색 가능. 벡터용으로만 증류 처리.

**④ 저장 — 단일 pgvector 테이블**

- 모든 소스가 **하나의 임베딩 테이블**에 안착(스키마 통일). 컬럼 예: `id, source_type, source_ref, content_raw, distilled, embedding, ts_last_active, idf_meta, metadata(jsonb)`.
- **차원 선택(중요, 검증됨)**: Cerebras는 3072차원 사용. pgvector는 표준 `vector`가 HNSW에서 **2000차원 한계** → 3072를 쓰려면 **`halfvec`(16bit, 4000차원까지)** 필수. 개인 PoC에선 저장·복잡도 절감을 위해 **voyage-code-3를 1024/2048차원 + int8 양자화(Matryoshka)로 쓰거나, 오픈웨이트 Qwen3-Embedding**를 권장 → 표준 `vector`로도 인덱싱 가능. 코드용/텍스트용 임베딩을 분리하되 **같은 테이블·다른 컬럼 또는 source_type 구분**.
- HNSW 인덱스(`m`, `ef_construction` 빌드 시 / `ef_search` 40~200 질의 시). FTS는 GIN.

**⑤ 하이브리드 검색** — 어떤 단일 채점자도 홀로 신뢰하지 않음

- **전문검색(FTS)**: 에러 문자열·플래그명·함수명 등 정확 토큰. 리터럴 매칭이 최선의 증거일 때 어떤 의미유사도도 못 이기게.
- **임베딩 검색**: 다른 표현(paraphrase) 연결.
- **IDF**: 희귀 토큰 중심 짧은 메시지를 승격, "고마워요!" 류는 0으로 수렴.
- **시간 감쇠**: 최신 세션 우대(옛 인프라 설명하는 오래된 답을 눌러줌). **코드/이력 메모리에서 특히 중요** — 리팩터링 전 코드나 폐기된 접근을 최신이 이기게.

**⑥ 융합 · 재순위**

- **RRF**: `score(d) = Σ_L w/(k + rank_L(d))`, `w=1.0, k=60`. 평활 상수 60이 **합의 > 단일 강한 표**를 만든다.
- 소스 수준 중복제거 + 파일당 결과 상한 → 다양성 있는 top-20.
- **리랭커**로 0~10 점수화 후 top-10. 2026년 기준 **Voyage rerank-2.5 / rerank-2.5-lite**(instruction-following) 권장, 오픈웨이트면 Qwen3 계열.
- **맥락 복원**: 위키 섹션이면 인접 섹션, 코드면 함수 상위 파일 맥락, 이력이면 앞뒤 메시지를 다시 붙여 청킹이 끊은 전제·주의를 복원("Lost in the Middle" 완화).

**⑦ Planner → Executor → Synthesis + MCP**

- **Planner**(가벼운 LLM): 질의를 보고 어떤 도구·소스를 쓸지 결정.
- **Executor**: 도구 호출을 병렬 팬아웃 → 공통 증거 스키마(점수·최신성·소스힌트)로 정규화.
- **Synthesis**: 타입 지정 증거 + 원질의 → 인용·주의사항 포함 최종 답변.
- **MCP 노출**: 검색 기본기를 얇고 LLM-비의존 도구로 노출(`search`, `search_code`, `search_history`). 그러면 Claude Code 등 어떤 에이전트나 오케스트레이터가 되어 개인 워크플로에 바로 물릴 수 있음. **개인 스코프에서 이게 최대 실익** — 별도 UI 없이 이미 쓰는 에이전트가 프론트엔드가 됨.

---

## 4. PIKE-RAG를 얹을 수 있는가 — 검토 결과

**결론: 예. 아주 잘 맞물린다.** Cerebras는 "실전 운영 아키텍처"(어떻게 지저분한 데이터를 파이프라인으로 다루나), PIKE-RAG는 "능력 단계·지식 표현 이론"(어떤 질문까지 답할 수 있게 키우나)이라 **층위가 겹치지 않고 보완적**이다. 매핑:

| PIKE-RAG 개념 | Cerebras 대응/공백 | 개인 시스템에 얹는 방법 | 난이도 |
|---|---|---|---|
| **L0 다층 이질 지식베이스**(정보원층·코퍼스층·증류지식층 그래프) | Cerebras는 **평면(flat) 임베딩 테이블** — 명시적 계층 없음 | 평면 테이블에 `parent_ref`/레이어 태그를 더해 파일↔함수, 세션↔버스트, 문서↔섹션의 **다중 구체성 링크**를 그래프처럼 질의 | 중 |
| **지식 원자화**(원자적 질문을 지식 인덱스로) | Cerebras 증류의 "한 줄 질문" + 버스팅과 **거의 동일 발상** | 이미 하는 증류를 "청크당 여러 원자 질문"으로 확장 → 다중 구체성 인덱스 강화 | **낮음(즉시)** |
| **오토태깅**(구어 질의 ↔ 전문 용어 매핑) | Cerebras엔 없음 | 코드/에이전트 용어와 자연어 질의 간극을 태그로 연결(예: "그때 그 락 문제" ↔ `mutex deadlock`) | 낮음 |
| **지식 인지 과업 분해**(멀티홉 반복 검색-생성) | Cerebras Planner는 **1회 계획·병렬 팬아웃** — 반복 분해 아님 | Planner를 **반복 루프**로 승격: 원자 질문 후보 검색→선택→맥락 누적→재질의. 멀티홉("A를 고친 커밋이 참조한 이슈의 원인은?")에 효과 | 중~높 |
| **L1~L4 능력 단계** | Cerebras는 단일 성숙 시스템 | **그대로 로드맵 축으로 채택**(아래 5장) | — |
| **학습형 분해기**(도메인 특화 파인튜닝) | 없음 | 개인 스코프 과투자. **보류**(데이터·평가셋 쌓인 뒤에만) | 높음(후순위) |

**얹지 말아야 할 것**: PIKE-RAG의 VLM 파일 파싱(멀티모달 표·차트), 예측(L3)·창작(L4)용 멀티에이전트 플래닝은 개인 코드/이력 메모리엔 과함. **L0~L2 개념만 취하고 L3~L4는 개념적 여지로만** 남긴다.

---

## 5. 로드맵 (Cerebras 파이프라인 × PIKE-RAG L0~L2 단계)

각 단계는 **그 자체로 동작하는 시스템**이고, 다음 단계가 하위 모듈을 상속·증강한다(PIKE-RAG 방식).

### Phase 0 — 뼈대: 단일 테이블 + 코드 인덱싱 (PIKE-RAG L0)
- Postgres + pgvector + FTS(GIN) 1개 테이블 스키마 확정.
- 임베딩 모델 선택 확정(권장: 코드 voyage-code-3@1024 또는 Qwen3-Embedding; 차원≤2000이면 표준 `vector`, 3072 고수 시 `halfvec`).
- CocoIndex로 코드 저장소 1~2개 증분 인덱싱.
- **완료 기준**: "이 함수 어디서 정의됐지?" 류 코드 검색이 grep보다 의미적으로 나은 결과.

### Phase 1 — 에이전트 메모리 수집 + 증류 (PIKE-RAG L1 factual)
- 이력 파서(Claude Code JSONL / SQLite → 표준 행) + 세션=스레드 재구성.
- **LLM 증류**(한줄질문·요약·해결책·참조) + **버스팅**(도구성공/재질문 가중).
- 하이브리드 검색(FTS+임베딩+IDF+시간감쇠) 가동.
- **완료 기준**: "지난달 그 빌드 에러 어떻게 풀었지?"에 정확한 과거 세션을 소환.

### Phase 2 — 융합·재순위·MCP (Cerebras 완성형)
- RRF(k=60) 융합 + 소스 중복제거 + 리랭커(rerank-2.5) top-10 + 맥락 복원.
- Planner→Executor→Synthesis 파이프라인.
- **MCP 얇은 도구**(`search`, `search_code`, `search_history`) 노출 → Claude Code에서 바로 호출.
- **완료 기준**: 코드+이력을 아우른 질의가 인용·주의사항 포함 답변으로 나오고, 에이전트가 도구로 호출 가능.

### Phase 3 — 다층화 + 원자화 + 오토태깅 (PIKE-RAG L1 강화)
- 평면 테이블에 레이어/부모 링크 추가(파일↔함수, 세션↔버스트, 문서↔섹션).
- **지식 원자화**(청크당 다중 원자 질문) + 오토태깅(구어↔전문용어).
- 프로젝트/스코프 개념 도입(코드 vs 메모리 vs 후순위 소스 분리 검색).
- **완료 기준**: 다중 구체성 검색으로 recall·precision 동시 개선(간이 평가셋으로 측정).

### Phase 4 — 멀티홉 분해 + 후순위 소스 (PIKE-RAG L2 reasoning)
- Planner를 **반복 검색-생성 루프**로 승격(지식 인지 과업 분해).
- 후순위 커넥터 추가: n8n → Gmail → GDrive → Slack → 위키(각각 플러그인 스크립트).
- (선택) 간이 평가셋 축적 후에만 학습형 분해기 검토.
- **완료 기준**: "A를 고친 세션이 참조한 이슈의 근본 원인" 같은 2~3홉 질의 응답.

---

## 6. 기술 스택 요약 (개인/PoC 권장값)

| 계층 | 권장 | 비고(검증됨) |
|---|---|---|
| 저장 | PostgreSQL + pgvector | HNSW + GIN(FTS)를 한 DB에 |
| 벡터 타입 | 표준 `vector`(≤2000d) / 3072d면 `halfvec` | halfvec=16bit·4000d·저장 절반, ef_search 40~200 |
| 코드 임베딩 | voyage-code-3(1024/2048, int8) 또는 Qwen3-Embedding | Matryoshka·양자화로 저장·비용 절감 |
| 텍스트 임베딩 | Gemini Embedding 2(최상급) 또는 Qwen3-Embedding-8B(오픈웨이트 MTEB 선두) | 개인이면 오픈웨이트로 로컬 실행 가능 |
| 코드 인덱서 | CocoIndex(Tree-sitter, 증분) | Postgres 리니지 추적, 변경 청크만 재계산 |
| 리랭커 | Voyage rerank-2.5 / -lite | instruction-following |
| 융합 | RRF k=60, w=1.0 | 합의 우선 |
| 노출 | MCP 얇은 도구 | Claude Code·MCP 에이전트가 오케스트레이터 |

---

## 7. 검증된 사실 (2026-07 웹 확인)

- pgvector 표준 `vector`는 HNSW **2000차원 한계** → 3072차원(Cerebras가 쓴 크기)은 **`halfvec`**로 인덱싱(4000차원까지, 저장 절반, ef_search 40~200 권장).
- **voyage-code-3**: 코드 검색 특화, 2048/1024/512/256차원 지원, Matryoshka + int8/binary 양자화로 저장·검색 비용 대폭 절감(OpenAI v3-large 대비 코드 검색 +13.8%).
- **Voyage rerank-2.5 / rerank-2.5-lite**: instruction-following 리랭커(현세대).
- 오픈웨이트 임베딩: **Qwen3-Embedding-8B**가 MTEB v2 오픈웨이트 선두권(~75%), **Gemini Embedding 2**가 종합 최상급으로 평가.
- **CocoIndex**(2026 활성): Tree-sitter 구문 청킹 + 증분(변경 파일만) + Postgres 리니지. "long-horizon 에이전트용 증분 엔진"으로 포지셔닝.
- **Claude Code 이력**: `~/.claude/projects/<encoded-path>/<session-id>.jsonl`, 줄당 JSON(type·content·timestamp·sessionId·cwd), 전역 `history.jsonl` + `sessions-index.json`(요약·메시지수·git브랜치). **기본 30일 후 삭제**(`cleanupPeriodDays`) → KB 적재 시 영구 메모리화.
- 에이전트 메모리 표준 분류: semantic·episodic·procedural·working → 본 시스템은 코드=semantic/procedural, 에이전트 이력=episodic/procedural에 대응.

---

## 8. 열린 결정 사항 (다음 단계에서 정할 것)

1. 임베딩을 **로컬(Qwen3, 비용 0·프라이버시)** vs **API(Voyage/Gemini, 품질·편의)** 중 무엇으로? — 개인 이력은 프라이버시상 로컬이 유리할 수 있음.
2. 3072차원(Cerebras 그대로) 고수 vs 1024~2048로 축소(권장). 후자면 halfvec 불필요.
3. 에이전트 이력 소스 목록 확정(Claude Code 외 Hermes의 실제 저장 포맷 샘플 필요).
4. 후순위 소스 착수 순서 확정(현재 가정: n8n → Gmail → GDrive → Slack → 위키).