# 구현 스펙: MCP 서버 도커 서빙 + SSE 트랜스포트 (Issue #10)

## 요구사항

MCP 서버를 도커 컨테이너로 서빙하고 클라이언트가 SSE로 접속. stdio는 로컬 dev 기본값으로 유지.

## 대상 파일

- 수정: `src/mcp_server.py`, `docker-compose.yml`
- 신규: `Dockerfile`, `.dockerignore`, `tests/test_mcp_transport.py`
- 그 외 src 파일 수정 금지.

## 변경 1 — `src/mcp_server.py` 트랜스포트 선택

- 순수 함수 추가:
  ```python
  def resolve_transport(env: Mapping[str, str]) -> tuple[str, str, int]:
      """(transport, host, port). MCP_TRANSPORT: stdio(기본)|sse|streamable-http.
      MCP_HOST 기본 "0.0.0.0", MCP_PORT 기본 8765. 잘못된 transport 값이면 ValueError."""
  ```
- `__main__`: `resolve_transport(os.environ)` 결과로 분기 —
  stdio면 기존 그대로 `mcp.run(transport="stdio")`,
  아니면 `FastMCP` 인스턴스에 host/port 설정 후 `mcp.run(transport=<값>)`.
  주의: FastMCP는 생성자/settings로 host·port를 받는다(`FastMCP("memory-base", host=..., port=...)`).
  env 파싱이 모듈 import 시가 아니라 `__main__`에서만 일어나도록 할 것(테스트 용이성).
- 파일 상단 docstring의 등록 예시를 두 가지로 갱신:
  - stdio(로컬): 기존 유지
  - SSE(도커): `claude mcp add --transport sse memory-base http://localhost:8765/sse`

## 변경 2 — `Dockerfile`

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/ src/
ENV MCP_TRANSPORT=sse
EXPOSE 8765
CMD ["uv", "run", "--no-sync", "python", "src/mcp_server.py"]
```
(정확한 uv 이미지 태그·옵션은 구현 시 동작 확인 후 조정 가능. 원칙: lock 기반 재현 빌드, dev 의존성 제외)

`.dockerignore`: `.venv`, `.git`, `.cocoindex`, `__pycache__`, `tests`, `docs`, `.env`

## 변경 3 — `docker-compose.yml`에 `mcp` 서비스 추가

```yaml
  mcp:
    build: .
    container_name: memory_base_mcp
    depends_on: [db]
    ports:
      - "8765:8765"
    environment:
      MCP_TRANSPORT: sse
      DB_URL: postgres://memory:memory@db:5432/memory_base   # 컨테이너 내부는 db:5432
      EMB_URL: ${EMB_URL}
      RERANK_URL: ${RERANK_URL}
      LLM_URL: ${LLM_URL}
```
기존 `db` 서비스는 수정 금지(볼륨·포트 유지).

## TDD — 테스트 먼저

`tests/test_mcp_transport.py` (DB/네트워크 무의존, 먼저 작성해 red 확인 후 구현):
- 기본값: 빈 env → ("stdio", "0.0.0.0", 8765)
- MCP_TRANSPORT=sse → ("sse", ...), streamable-http 허용, 대소문자 무시
- MCP_PORT=9000 → port 9000 (int), MCP_HOST 반영
- 잘못된 transport("tcp") → ValueError
- MCP_PORT 비숫자 → ValueError

## 수용 기준

1. `uv run pytest tests/test_mcp_transport.py -q` red→green, 기존 테스트 전부 유지
2. (오케스트레이터) `docker compose up -d --build mcp` → SSE 엔드포인트(`curl http://localhost:8765/sse`)가 이벤트 스트림 응답
3. (오케스트레이터) MCP 클라이언트로 SSE 접속해 tools/list + search_code 실호출 성공

## 비고

- MCP 프로토콜상 SSE는 legacy이고 streamable-http가 현행 표준이지만 **사용자가 SSE를 명시 요청** —
  SSE를 기본으로 하되 env로 streamable-http도 선택 가능하게(한 줄 차이).
- 컨테이너에서 외부 vLLM(192.168.x.x)은 브리지 네트워크로 접근 가능.
