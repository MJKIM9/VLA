---
name: weekly-report
description: 이번 기간의 개발 내용을 git 기록과 코드 분석으로 정리해 진행 보고서(상세본)와 발표용 보고를 자동 생성한다. 팀원이 /weekly-report 로 호출한다.
command: /weekly-report
allowed-tools: Bash(git log:*), Bash(git diff:*), Bash(git status:*), Bash(git init:*), Bash(git add:*), Bash(git commit:*), Bash(ls:*), Bash(find:*), Bash(mkdir:*), Read, Grep, Glob, Write
---

# 진행 보고서 자동 생성

이번 기간에 개발한 내용을 상세히 담은 보고서 2종(상세본, 발표용)을 자동으로 만든다.
사람이 손으로 채우는 항목은 없다. 전체를 Claude Code가 작성한다.

## 시작 전: 공통 지침 확인 (필수)
보고서를 만들기 전에, 이 워크스페이스의 `CLAUDE.md` 를 Read 로 먼저 읽는다.
그 안의 "자동화팀 공통 지침" 블록(마커로 감싼 영역)에 적힌 규칙을 이번 실행에 그대로 적용한다.
특히 보고서 구성, 문서 표기 규칙(가운뎃점 대신 쉼표 또는 '및'), git 및 코드 업로드 규칙을 확인한 뒤 진행한다.
`CLAUDE.md` 가 없거나 공통 지침 블록이 없으면, 먼저 설치 스크립트(weekly_report_setup.sh)를 실행하라고 사용자에게 안내하고 멈춘다.

## 0단계: git repo 확인
`git status` 로 저장소 여부를 확인한다. git 저장소가 아니면 아래를 안내하고 초기화한다.
```
git init
git add -A
git commit -m "chore: 초기 커밋"
```

## 1단계: 근거 수집 (git)
- `git log --since="7 days ago" --oneline`
- `git diff --stat "@{7.days.ago}" HEAD` (실패하면 `git diff --stat HEAD~5 HEAD`)
이 출력을 이번 기간 작업의 기본 근거로 삼는다. 커밋이 없으면 지어내지 말고 "이번 기간 커밋 없음"으로 적는다.

## 2단계: 개발 내용 분석
git 근거로 파악한 변경 지점을 중심으로, 개발 내용을 설명하는 데 필요한 범위에서 코드를 분석한다.
다음을 파악한다.
- 원리: 무엇을 어떤 방식과 원리로 구현했는가
- 구성: 모듈, 클래스, 파일 구성
- 순서 및 단계: 처리 흐름 또는 개발 단계
- 패키지 구성: 디렉터리 및 패키지 구조
코드와 커밋에서 확인되는 내용만 적는다. 확인되지 않는 것은 지어내지 않는다.

## 3단계: 상세본 생성
1. `WEEKLY_REPORT_TEMPLATE.md` 를 읽는다.
2. 모든 항목을 자동으로 채운다. 작성일은 오늘 날짜, 담당자는 git 사용자 이름.
3. "2. 개발 내용 상세"에 원리, 구성, 순서 및 단계, 패키지 구성을 서술하고, "3. 변경 근거"에 git 로그와 diff --stat 출력을 붙인다.
4. `reports/WEEKLY_YYYY-MM-DD.md` 로 저장한다. (`mkdir -p reports` 먼저)

## 4단계: 발표용 생성
1. `WEEKLY_BRIEFING_TEMPLATE.md` 를 읽는다.
2. 상세본을 요약해 핵심 개발 내용 중심으로 자동 작성한다.
3. `reports/BRIEFING_YYYY-MM-DD.md` 로 저장한다.

## 5단계: 보고서 커밋
1. 생성한 `reports/` 파일을 `git add reports/` 로 추가한다.
2. `진행 보고 자동 생성 YYYY-MM-DD` 형식의 한글 메시지로 커밋한다.

## 6단계: 코드 업로드
1. 아래 명령을 그대로 실행한다. 직접 git add/commit/push 로 대신하지 말고, 업로드 폴더 이름을 임의로 짓지 마라.
   `bash ~/automation_git_package/3_code_upload/code_sync.sh`
2. 이 스크립트가 개인 설정(config.env)에 맞는 업로드 폴더(예: `koras_automation_code_upload_<이름>`)를 스스로 고른다. 그 폴더로 현재 워크스페이스의 src 를 복사하고 커밋, 푸시한다. 코드를 따로 올릴 필요는 없다.
3. 스크립트가 없거나 오류가 나면 그대로 사용자에게 알린다. 다른 폴더로 임의 업로드하지 마라.

## 규칙
- 사람이 채우는 항목을 남기지 않는다. 전체를 자동으로 작성한다.
- 코드와 git에서 확인되지 않는 내용은 넣지 않는다.
- 원본 템플릿 2개는 수정하지 않는다.
- 문서에 가운뎃점을 쓰지 않고 쉼표 또는 '및'으로 표기한다.
- 마지막에 저장한 파일 2개 경로만 보고하고 끝낸다.
