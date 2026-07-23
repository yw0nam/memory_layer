# Deployment Overview

Project Lantern is deployed as two application replicas behind a health-checking reverse proxy. A release is considered healthy when both replicas report the new version, the five-minute error rate remains below one percent, and the queue depth returns to its normal range. Operators keep the previous container image for one release so rollback does not depend on rebuilding an older commit.

## Korean Operator Notes

배포 중 오류율이 1퍼센트를 넘으면 새 컨테이너로 보내는 트래픽을 즉시 중단한다. 이전 이미지의 두 복제본을 다시 시작하고 상태 확인이 통과할 때까지 외부 요청을 받지 않는다. 롤백 뒤에는 대기열 깊이와 데이터베이스 연결 수를 10분 동안 관찰한다. 장애 기록에는 배포 버전, 최초 경고 시각, 롤백 완료 시각, 담당자 이름을 남긴다. 지연된 작업이 정상적으로 처리되는지 확인하고 고객 지원팀에 현재 상태와 다음 점검 시간을 공유한다.

## Verification Commands

After rollback, operators query the `/health/version` endpoint on each replica and compare the reported image digest with the release manifest. They also inspect the worker queue for jobs older than five minutes and run the synthetic checkout probe from two regions. The incident closes only after the error rate remains below one percent for ten consecutive minutes and delayed jobs begin completing normally.
