# AI 브랜드 아이덴티티 생성기

브랜드 브리프 JSON 하나로 네이밍, 슬로건, 브랜드 스토리, 컬러 팔레트와
로고 시안을 생성하는 대화형 Python CLI 프로젝트입니다.

## 주요 기능

- 브랜드명 후보 4개와 의미/유래 생성
- 브랜드 톤앤매너에 맞는 슬로건 3개 생성
- 탄생 배경·철학·비전을 포함한 300자 이내 브랜드 스토리 생성
- 메인 컬러 1개와 서브 컬러 3개 추천 및 PNG 팔레트 저장
- 서로 다른 방향의 로고 시안 2개를 PNG로 저장
- 단계별 오류를 출력하고 가능한 다음 단계를 계속 수행
- 모든 텍스트 결과와 오류 내역을 `brand_result.json`으로 저장

## 요구사항 대응표

| 미션 요구사항 | 구현 위치 |
|---|---|
| `print`/`input` 기반 브리프·출력 경로 입력 | `ask_brief_path`, `ask_output_dir` |
| 필수/선택 JSON 필드 검증 | `validate_brief` |
| 네이밍 3~5개와 의미 | `generate_naming` (4개) |
| 슬로건 3개 | `generate_slogans` |
| 300자 내외 스토리 | `generate_story`, `limit_story` |
| 메인 1개·서브 2~3개 컬러 | `generate_colors` (서브 3개) |
| 컬러 팔레트 PNG | `save_color_palette` |
| 로고 PNG 2~3개 | `generate_logos` (2개) |
| 단계별 API 오류 처리 | `api_error_message`, `main` |
| 환경변수 API 키 관리 | `get_client`, `.env.example` |

## 실행 환경

- Python 3.10 이상
- OpenAI API 키

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS/Linux:

```bash
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env`에 발급받은 키를 입력합니다. 실제 키가 들어간 `.env`는 Git에서 제외됩니다.

```dotenv
OPENAI_API_KEY=your_api_key_here
```

필요하면 `.env`에서 모델을 변경할 수 있습니다.

```dotenv
OPENAI_TEXT_MODEL=gpt-4o-mini
OPENAI_IMAGE_MODEL=gpt-image-2
```

## 브리프 작성

```json
{
  "industry": "친환경 화장품",
  "target": "20-30대 여성",
  "keywords": ["자연", "순수", "건강"],
  "tone": "따뜻하고 신뢰감 있는",
  "competitors": ["이니스프리", "아로마티카"],
  "notes": "도시에서도 지속 가능한 루틴을 강조"
}
```

- 필수: `industry`, `target`, `keywords`
- 선택: `tone`, `competitors`, `notes`

## 실행

```bash
python main.py
```

```text
AI 브랜드 아이덴티티 생성기
브리프 JSON 파일 경로를 입력하세요 (필수): brief.json
출력 폴더 경로를 입력하세요 (기본값: ./output):

[1/5] 브랜드 네이밍 생성 중...
[2/5] 슬로건 생성 중...
[3/5] 브랜드 스토리 생성 중...
[4/5] 컬러 팔레트 생성 중...
[5/5] 로고 시안 생성 중...
```

출력 폴더 구조:

```text
output/
├── brand_result.json
├── color_palette.png
├── logo_01.png
└── logo_02.png
```

## 테스트

테스트에서는 유료 API를 호출하지 않습니다.

```bash
python -m unittest discover -s tests -v
```

## 오류 처리

- 브리프 경로나 JSON 형식이 잘못되면 원인을 출력하고 종료합니다.
- API 키가 없거나 인증에 실패하면 `.env` 확인 안내를 출력합니다.
- 텍스트 생성 단계가 실패해도 가능한 다음 단계와 결과 JSON 저장을 계속합니다.
- 로고 한 시안이 실패해도 나머지 시안 생성을 계속합니다.
- 발생한 오류는 터미널과 `brand_result.json`의 `errors` 필드에서 확인할 수 있습니다.

## 보안

API 키를 `main.py`, `brief.json`, README 또는 커밋에 입력하지 마세요. 키가 노출되면
즉시 폐기하고 새 키를 발급해야 합니다.
