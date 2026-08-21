"""AI 브랜드 아이덴티티 생성기.

브랜드 브리프(JSON)를 입력받아 OpenAI API로 브랜드 네이밍, 슬로건, 스토리,
컬러 팔레트, 로고 시안까지 전부 생성해 출력 폴더에 저장하는 단일 CLI 프로그램.

사용법:
    1) .env 파일에 OPENAI_API_KEY=sk-... 를 넣는다
    2) pip install openai python-dotenv matplotlib --break-system-packages
    3) python main.py
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv
from openai import OpenAI

NAMING_MODEL = "gpt-4o"
COLOR_MODEL = "gpt-5-mini"
IMAGE_MODEL = "gpt-image-1"
REQUIRED_BRIEF_FIELDS = ("industry", "target", "keywords")
HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
MAX_STORY_CHARS = 300
LOGO_COUNT = 2
KOREAN_FONT_CANDIDATES = (
    "Malgun Gothic",
    "AppleGothic",
    "NanumGothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
)


# ── 브리프 로드 ──────────────────────────────────────────────


def load_json(path: str | Path) -> dict[str, Any]:
    """UTF-8 JSON 파일을 읽는다."""
    path = Path(path)
    if path.is_dir():
        raise ValueError(f"'{path}'는 폴더입니다. JSON 파일 경로를 입력하세요.")
    if not path.exists():
        raise ValueError(f"파일을 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON의 최상위 값은 객체여야 합니다: {path}")
    return data


def validate_brief(brief: dict[str, Any]) -> None:
    """브리프의 필수 필드를 검사한다."""
    missing = [field for field in REQUIRED_BRIEF_FIELDS if not brief.get(field)]
    if missing:
        raise ValueError(f"브리프 필수 필드가 없습니다: {', '.join(missing)}")
    if not isinstance(brief["keywords"], list):
        raise ValueError("브리프의 keywords는 문자열 배열이어야 합니다.")


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """결과를 읽기 쉬운 UTF-8 JSON으로 저장한다."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# ── OpenAI 클라이언트 및 JSON 채팅 호출 ──────────────────────


def get_client() -> OpenAI:
    """환경변수에서 API 키를 읽어 OpenAI 클라이언트를 반환한다."""
    load_dotenv()
    return OpenAI()


def call_json_chat(client: OpenAI, system: str, prompt: str, model: str = NAMING_MODEL) -> dict:
    """시스템/유저 프롬프트로 채팅을 호출하고 JSON 객체로 파싱해 반환한다."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
        )
    except Exception as e:
        raise RuntimeError(f"API 호출 중 오류가 발생했습니다: {e}") from e

    raw_text = response.choices[0].message.content
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"응답을 JSON으로 파싱하지 못했습니다. 원본 응답: {raw_text}") from e


def extract_json(text: str) -> dict[str, Any]:
    """모델 응답에서 JSON 객체를 추출한다."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("AI 응답에서 JSON을 찾지 못했습니다.")
        result = json.loads(cleaned[start : end + 1])

    if not isinstance(result, dict):
        raise ValueError("AI 응답의 최상위 값이 JSON 객체가 아닙니다.")
    return result


# ── 브랜드 네이밍 ────────────────────────────────────────────


def build_naming_prompt(brief: dict) -> str:
    """브리프 내용을 바탕으로 네이밍 프롬프트를 구성한다."""
    tone = brief.get("tone", "특별히 지정되지 않음")
    notes = brief.get("notes", "")

    return f"""
다음 브랜드 브리프를 바탕으로 브랜드 네이밍 후보를 생성해줘.

- 업종: {brief['industry']}
- 타겟: {brief['target']}
- 키워드: {', '.join(brief['keywords'])}
- 톤앤매너: {tone}
- 추가 요청사항: {notes if notes else '없음'}

요구사항:
- 브랜드명 후보 3~5개를 생성한다.
- 각 후보마다 이름의 의미나 유래를 1~2문장으로 설명한다.
- 결과는 반드시 아래 JSON 형식으로만 응답한다. 다른 설명이나 텍스트는 절대 포함하지 않는다.

{{
  "naming_candidates": [
    {{"name": "브랜드명1", "meaning": "의미 설명1"}},
    {{"name": "브랜드명2", "meaning": "의미 설명2"}}
  ]
}}
""".strip()


def generate_naming(client: OpenAI, brief: dict) -> dict:
    """OpenAI API를 호출해 브랜드 네이밍 후보를 생성한다."""
    prompt = build_naming_prompt(brief)
    return call_json_chat(client, "너는 브랜드 네이밍 전문가다. 반드시 JSON 형식으로만 응답한다.", prompt)


# ── 슬로건 ───────────────────────────────────────────────────


def build_slogan_prompt(brief: dict, brand_name: str, meaning: str) -> str:
    """선정된 브랜드명을 바탕으로 슬로건 프롬프트를 구성한다."""
    tone = brief.get("tone", "특별히 지정되지 않음")

    return f"""
다음 브랜드를 위한 홍보 슬로건 후보를 생성해줘.

- 브랜드명: {brand_name}
- 브랜드명의 의미: {meaning}
- 업종: {brief['industry']}
- 타겟: {brief['target']}
- 키워드: {', '.join(brief['keywords'])}
- 톤앤매너: {tone}

요구사항:
- 브랜드명과 자연스럽게 어울리면서 임팩트 있고 홍보 효과가 높은 한국어 슬로건 3개를 만든다.
- 각 슬로건은 15자 내외로 짧고 강렬하게 작성한다.
- 각 슬로건마다 왜 효과적인지 1문장으로 이유를 설명한다.
- 결과는 반드시 아래 JSON 형식으로만 응답한다. 다른 설명이나 텍스트는 절대 포함하지 않는다.

{{
  "slogans": [
    {{"text": "슬로건1", "reason": "이유1"}},
    {{"text": "슬로건2", "reason": "이유2"}}
  ]
}}
""".strip()


def generate_slogan(client: OpenAI, brief: dict, brand_name: str, meaning: str) -> dict:
    """선정된 브랜드명을 기준으로 슬로건 후보를 생성한다."""
    prompt = build_slogan_prompt(brief, brand_name, meaning)
    return call_json_chat(client, "너는 카피라이팅 전문가다. 반드시 JSON 형식으로만 응답한다.", prompt)


# ── 브랜드 스토리 (300자 제약 자동 강제) ─────────────────────


def build_story_prompt(brief: dict, brand_name: str, meaning: str, slogan: str) -> str:
    """선정된 슬로건을 바탕으로 브랜드 스토리 프롬프트를 구성한다."""
    tone = brief.get("tone", "특별히 지정되지 않음")

    return f"""
다음 브랜드의 슬로건을 중심 메시지로 삼아 브랜드 스토리를 작성해줘.

- 브랜드명: {brand_name}
- 브랜드명의 의미: {meaning}
- 슬로건: {slogan}
- 업종: {brief['industry']}
- 타겟: {brief['target']}
- 키워드: {', '.join(brief['keywords'])}
- 톤앤매너: {tone}

요구사항:
- 슬로건이 전하는 메시지를 스토리의 핵심 줄기로 삼는다.
- 한국어로 약 300자 분량으로 작성하며 탄생 배경, 철학, 비전을 포함한다.
- 결과는 반드시 아래 JSON 형식으로만 응답한다. 다른 설명이나 텍스트는 절대 포함하지 않는다.

{{
  "story": "브랜드 스토리"
}}
""".strip()


def generate_story(client: OpenAI, brief: dict, brand_name: str, meaning: str, slogan: str) -> dict:
    """선정된 슬로건을 기준으로 브랜드 스토리를 생성한다."""
    prompt = build_story_prompt(brief, brand_name, meaning, slogan)
    return call_json_chat(client, "너는 브랜드 스토리텔링 전문가다. 반드시 JSON 형식으로만 응답한다.", prompt)


def build_story_shorten_prompt(brand_name: str, slogan: str, story: str, max_chars: int) -> str:
    """길이 제한을 넘긴 스토리를 축약하기 위한 프롬프트를 구성한다."""
    return f"""
아래 브랜드 스토리는 공백 포함 {len(story)}자로 {max_chars}자 제한을 초과했다.
핵심 내용(탄생 배경, 철학, 비전)은 유지하면서 반드시 {max_chars}자 이내로 압축해줘.

- 브랜드명: {brand_name}
- 슬로건: {slogan}
- 현재 스토리: {story}

요구사항:
- 공백을 포함한 전체 글자 수가 {max_chars}자를 절대 넘지 않아야 한다.
- 문장이 중간에 끊기지 않고 자연스럽게 마무리되어야 한다.
- 결과는 반드시 아래 JSON 형식으로만 응답한다. 다른 설명이나 텍스트는 절대 포함하지 않는다.

{{
  "story": "축약된 브랜드 스토리"
}}
""".strip()


def trim_story_to_limit(story: str, max_chars: int) -> str:
    """마지막 수단으로 문장 경계에 맞춰 스토리를 강제로 잘라낸다."""
    if len(story) <= max_chars:
        return story
    truncated = story[:max_chars]
    last_sentence_end = max(truncated.rfind(mark) for mark in (".", "!", "?"))
    if last_sentence_end > 0:
        return truncated[: last_sentence_end + 1]
    return truncated


def generate_story_within_limit(
    client: OpenAI,
    brief: dict,
    brand_name: str,
    meaning: str,
    slogan: str,
    max_chars: int = MAX_STORY_CHARS,
    max_retries: int = 3,
) -> str:
    """스토리를 생성하고, 글자 수 제약(기본 300자)을 넘으면 자동으로 축약한다."""
    story_result = generate_story(client, brief, brand_name, meaning, slogan)
    story = story_result.get("story", "").strip()

    attempt = 0
    while len(story) > max_chars and attempt < max_retries:
        attempt += 1
        print(
            f"    ! 스토리가 {len(story)}자로 {max_chars}자 제한을 초과했습니다. "
            f"축약 재생성 중... ({attempt}/{max_retries})"
        )
        prompt = build_story_shorten_prompt(brand_name, slogan, story, max_chars)
        result = call_json_chat(client, "너는 브랜드 스토리텔링 전문가다. 반드시 JSON 형식으로만 응답한다.", prompt)
        story = result.get("story", "").strip()

    if len(story) > max_chars:
        print(f"    ! 재생성 후에도 {len(story)}자로 제한을 초과하여 문장 단위로 잘라냅니다.")
        story = trim_story_to_limit(story, max_chars)

    return story


# ── 컬러 팔레트 ──────────────────────────────────────────────


def generate_colors(client: OpenAI, brief: dict, brand_name: str, slogan: str, story: str) -> dict:
    """브랜드명·슬로건·스토리를 바탕으로 컬러 팔레트를 JSON으로 생성한다."""
    prompt = f"""
당신은 B2B 커피 브랜드 전문 브랜드 전략가입니다.
아래 브랜드 정보를 바탕으로 어울리는 컬러 팔레트를 제안하세요.

[브랜드 브리프]
{json.dumps(brief, ensure_ascii=False, indent=2)}

[브랜드명]
{brand_name}

[슬로건]
{slogan}

[브랜드 스토리]
{story}

요구사항:
1. main_color는 1개, sub_colors는 2~3개를 제안합니다.
2. 모든 색은 name, hex, reason을 포함하며 hex는 #RRGGBB 형식이어야 합니다.
3. 슬로건과 스토리의 톤, 전문성, 신뢰감, 프리미엄 원두, 천연 딸기향의 차별성을 표현합니다.
4. 설명 문장이나 마크다운 없이 아래 구조의 JSON 객체만 반환하세요.

{{
  "colors": {{
    "main_color": {{"name": "색상명", "hex": "#000000", "reason": "선정 이유"}},
    "sub_colors": [
      {{"name": "색상명", "hex": "#FFFFFF", "reason": "선정 이유"}}
    ]
  }}
}}
""".strip()

    response = client.responses.create(model=COLOR_MODEL, input=prompt)
    result = extract_json(response.output_text)
    validate_colors(result.get("colors"))
    return result


def validate_colors(colors: Any) -> None:
    """팔레트 구조와 HEX 값을 검사하고 HEX를 대문자로 정규화한다."""
    if not isinstance(colors, dict):
        raise ValueError("colors가 JSON 객체가 아닙니다.")

    main_color = colors.get("main_color")
    sub_colors = colors.get("sub_colors")
    if not isinstance(main_color, dict) or not isinstance(sub_colors, list):
        raise ValueError("main_color 또는 sub_colors 형식이 올바르지 않습니다.")
    if not 2 <= len(sub_colors) <= 3:
        raise ValueError("서브 컬러는 2~3개여야 합니다.")

    for color in [main_color, *sub_colors]:
        hex_value = str(color.get("hex", "")) if isinstance(color, dict) else ""
        if not isinstance(color, dict) or not HEX_PATTERN.fullmatch(hex_value):
            raise ValueError(f"올바르지 않은 HEX 컬러입니다: {color}")
        color["hex"] = hex_value.upper()


def setup_korean_font() -> None:
    """설치된 폰트 중 한글을 지원하는 폰트를 찾아 matplotlib에 적용한다."""
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in KOREAN_FONT_CANDIDATES:
        if candidate in installed:
            matplotlib.rcParams["font.family"] = candidate
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def readable_text_color(hex_color: str) -> str:
    """배경 HEX 색상의 밝기에 따라 대비되는 텍스트 색을 고른다."""
    red, green, blue = (int(hex_color[index : index + 2], 16) for index in (1, 3, 5))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#222222" if luminance > 150 else "white"


def wrap_text(text: str, max_chars: int = 22) -> str:
    """긴 설명 문장을 여러 줄로 감싼다."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def save_color_palette(colors: dict[str, Any], output_path: Path, brief: dict[str, Any] | None = None) -> None:
    """추천 컬러를 라벨·HEX·이유가 담긴 한 장의 PNG 이미지로 저장한다."""
    setup_korean_font()

    palette = [colors["main_color"], *colors["sub_colors"]]
    figure, axes = plt.subplots(1, len(palette), figsize=(3.2 * len(palette), 6))
    if len(palette) == 1:
        axes = [axes]

    industry = (brief or {}).get("industry", "Brand")
    figure.suptitle(f"Brand Color Palette  |  {industry}", fontsize=15, fontweight="bold", y=0.98)

    for index, (axis, color) in enumerate(zip(axes, palette)):
        label = "MAIN" if index == 0 else f"SUB {index}"
        text_color = readable_text_color(color["hex"])
        axis.add_patch(
            mpatches.FancyBboxPatch(
                (0.05, 0.42),
                0.9,
                0.5,
                boxstyle="round,pad=0.02",
                facecolor=color["hex"],
                edgecolor="white",
                linewidth=2,
            )
        )
        axis.text(0.5, 0.83, label, ha="center", va="center", fontsize=10,
                   fontweight="bold", color=text_color, transform=axis.transAxes)
        axis.text(0.5, 0.59, color["hex"], ha="center", va="center", fontsize=12,
                   fontweight="bold", color=text_color, transform=axis.transAxes)
        axis.text(0.5, 0.34, color["name"], ha="center", va="top", fontsize=9,
                   fontweight="bold", color="#333333", transform=axis.transAxes)
        axis.text(0.5, 0.26, wrap_text(color.get("reason", "")), ha="center", va="top",
                   fontsize=7.5, color="#666666", transform=axis.transAxes, multialignment="center")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# ── 로고 시안 ────────────────────────────────────────────────


def build_logo_prompt(brief: dict[str, Any], content_result: dict[str, Any], concept_number: int) -> str:
    """서로 다른 로고 시안을 위한 이미지 프롬프트를 만든다."""
    colors = content_result["colors"]
    color_text = ", ".join(
        f'{color["name"]} {color["hex"]}' for color in [colors["main_color"], *colors["sub_colors"]]
    )
    concept_styles = {
        1: "minimal geometric symbol combining a coffee bean and a subtle strawberry motif",
        2: "refined wordmark with a small abstract aroma curve and coffee bean emblem",
        3: "premium circular seal symbol suitable for wholesale coffee packaging",
    }

    return f"""
Create one professional B2B coffee brand logo concept.
Brand name: {content_result['brand_name']}
Slogan: {content_result.get('slogan', '')}
Industry: {brief['industry']}
Target: {brief['target']}
Keywords: {', '.join(map(str, brief['keywords']))}
Tone: {brief.get('tone', 'professional, trustworthy, premium')}
Approved colors: {color_text}
Concept direction: {concept_styles[concept_number]}
Use a clean flat vector-like style, centered composition, plain light background,
strong silhouette, limited approved color palette, no mockup, no photograph,
no package, no extra slogan, and no text except the exact brand name.
""".strip()


def generate_logos(
    client: OpenAI, brief: dict[str, Any], content_result: dict[str, Any], output_dir: Path, count: int
) -> list[str]:
    """로고 시안을 각각 PNG 파일로 저장한다."""
    saved_files: list[str] = []
    for number in range(1, count + 1):
        try:
            response = client.images.generate(
                model=IMAGE_MODEL,
                prompt=build_logo_prompt(brief, content_result, number),
                size="1024x1024",
            )
            image_base64 = response.data[0].b64_json
            if not image_base64:
                raise ValueError("이미지 API 응답에 이미지 데이터가 없습니다.")

            filename = f"logo_{number:02d}.png"
            (output_dir / filename).write_bytes(base64.b64decode(image_base64))
            saved_files.append(filename)
            print(f"    - 저장: {output_dir / filename}")
        except Exception as error:
            print(f"    ! 로고 시안 {number} 생성 실패: {error}")
    return saved_files


# ── 파이프라인 단계 ──────────────────────────────────────────


def ask_brief_path() -> str:
    """브리프 파일 경로를 입력받는다. 엔터만 치면 기본값 brief.json을 사용한다."""
    raw = input("브리프 파일 경로를 입력하세요 (엔터 시 brief.json): ").strip()
    return raw or "brief.json"


def ask_output_dir() -> Path:
    """선택 입력인 출력 폴더 경로를 입력받는다. 기본값은 ./output."""
    raw = input("출력 폴더 경로를 입력하세요 (엔터 시 ./output): ").strip()
    return Path(raw or "./output")


def step_naming(client: OpenAI, brief: dict[str, Any]) -> tuple[list, str, str]:
    """[1/5] 브랜드 네이밍 후보를 생성하고 첫 번째 후보를 선택한다."""
    print("\n[1/5] 브랜드 네이밍 생성 중...")
    try:
        result = generate_naming(client, brief)
        candidates = result.get("naming_candidates", [])
        if not candidates:
            raise RuntimeError("생성된 네이밍 후보가 없습니다.")
    except Exception as error:
        print(f"  ❌ 네이밍 생성 실패: {error}")
        fallback_name = f"{brief.get('industry', '브랜드')} 브랜드"
        return [], fallback_name, ""

    for item in candidates:
        print(f"  - {item.get('name', '(이름 없음)')}: {item.get('meaning', '')}")

    selected = candidates[0]
    return candidates, selected.get("name", ""), selected.get("meaning", "")


def step_slogan(client: OpenAI, brief: dict[str, Any], brand_name: str, meaning: str) -> tuple[list, str]:
    """[2/5] 선택된 브랜드명을 기준으로 슬로건 후보를 생성한다."""
    print("\n[2/5] 슬로건 생성 중...")
    try:
        result = generate_slogan(client, brief, brand_name, meaning)
        slogans = result.get("slogans", [])
        if not slogans:
            raise RuntimeError("생성된 슬로건이 없습니다.")
    except Exception as error:
        print(f"  ❌ 슬로건 생성 실패: {error}")
        return [], ""

    for item in slogans:
        print(f'  - "{item.get("text", "")}"')

    return slogans, slogans[0].get("text", "")


def step_story(client: OpenAI, brief: dict[str, Any], brand_name: str, meaning: str, slogan: str) -> str:
    """[3/5] 슬로건을 기반으로 브랜드 스토리를 생성하고 300자 제약을 강제한다."""
    print("\n[3/5] 브랜드 스토리 생성 중...")
    try:
        story = generate_story_within_limit(client, brief, brand_name, meaning, slogan)
        print(f"  - 스토리 생성 완료 ({len(story)}자, 제한 {MAX_STORY_CHARS}자)")
        return story
    except Exception as error:
        print(f"  ❌ 스토리 생성 실패: {error}")
        return ""


def step_colors(
    client: OpenAI, brief: dict[str, Any], output_dir: Path, brand_name: str, slogan: str, story: str
) -> dict[str, Any] | None:
    """[4/5] 컬러 팔레트를 생성하고 PNG로 시각화해 저장한다."""
    print("\n[4/5] 컬러 팔레트 생성 중...")
    try:
        result = generate_colors(client, brief, brand_name, slogan, story)
        colors = result["colors"]
    except Exception as error:
        print(f"  ❌ 컬러 팔레트 생성 실패: {error}")
        return None

    main_color = colors["main_color"]
    sub_hex = ", ".join(color["hex"] for color in colors["sub_colors"])
    print(f"  - 메인: {main_color['hex']} ({main_color['name']})")
    print(f"  - 서브: {sub_hex}")

    try:
        palette_path = output_dir / "color_palette.png"
        save_color_palette(colors, palette_path, brief)
        print(f"  - 저장: {palette_path}")
    except Exception as error:
        print(f"  ! 컬러 팔레트 이미지 저장 실패: {error}")

    return colors


def step_logos(
    client: OpenAI, brief: dict[str, Any], output_dir: Path, brand_name: str, slogan: str, colors: dict[str, Any] | None
) -> list[str]:
    """[5/5] 로고 시안을 이미지 생성 API로 만들어 PNG로 저장한다."""
    print("\n[5/5] 로고 시안 생성 중...")
    if colors is None:
        print("  ! 컬러 팔레트가 없어 로고 생성을 건너뜁니다.")
        return []

    content_result = {"brand_name": brand_name, "slogan": slogan, "colors": colors}
    return generate_logos(client, brief, content_result, output_dir, LOGO_COUNT)


def main() -> None:
    print("\n🎨 AI 브랜드 아이덴티티 생성기\n")

    brief_path = ask_brief_path()
    output_dir = ask_output_dir()

    try:
        brief = load_json(brief_path)
        validate_brief(brief)
    except (OSError, ValueError) as error:
        print(f"❌ 브리프 로드 실패: {error}")
        return

    try:
        client = get_client()
    except Exception as error:
        print(f"❌ OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요. ({error})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    candidates, brand_name, meaning = step_naming(client, brief)
    slogans, slogan = step_slogan(client, brief, brand_name, meaning)
    story = step_story(client, brief, brand_name, meaning, slogan)
    colors = step_colors(client, brief, output_dir, brand_name, slogan, story)
    logo_files = step_logos(client, brief, output_dir, brand_name, slogan, colors)

    result = {
        "naming_candidates": candidates,
        "brand_name": brand_name,
        "brand_meaning": meaning,
        "slogans": slogans,
        "slogan": slogan,
        "story": story,
        "colors": colors,
        "logo_files": logo_files,
    }
    save_json(result, output_dir / "brand_result.json")

    print(f"\n✅ 완료! {output_dir}/ 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
