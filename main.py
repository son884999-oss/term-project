"""대화형 AI 브랜드 아이덴티티 생성기."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI

DEFAULT_TEXT_MODEL = "gpt-4o-mini"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
REQUIRED_BRIEF_FIELDS = ("industry", "target", "keywords")
HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
STORY_MAX_CHARS = 300
LOGO_COUNT = 2


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise ValueError(f"브리프 JSON 파일을 찾을 수 없습니다: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"올바른 JSON 형식이 아닙니다: {path} ({error.msg})") from error
    if not isinstance(data, dict):
        raise ValueError("브리프 JSON의 최상위 값은 객체여야 합니다.")
    return data


def validate_brief(brief: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_BRIEF_FIELDS if not brief.get(field)]
    if missing:
        raise ValueError(f"필수 필드가 없습니다: {', '.join(missing)}")
    if not isinstance(brief["industry"], str) or not isinstance(brief["target"], str):
        raise ValueError("industry와 target은 문자열이어야 합니다.")
    if not isinstance(brief["keywords"], list) or not all(
        isinstance(keyword, str) and keyword.strip() for keyword in brief["keywords"]
    ):
        raise ValueError("keywords는 비어 있지 않은 문자열 배열이어야 합니다.")
    if "tone" in brief and not isinstance(brief["tone"], str):
        raise ValueError("tone은 문자열이어야 합니다.")
    if "notes" in brief and not isinstance(brief["notes"], str):
        raise ValueError("notes는 문자열이어야 합니다.")
    if "competitors" in brief and (
        not isinstance(brief["competitors"], list)
        or not all(isinstance(item, str) for item in brief["competitors"])
    ):
        raise ValueError("competitors는 문자열 배열이어야 합니다.")


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_client() -> OpenAI:
    """환경 변수 또는 로컬 .env에서만 API 키를 읽는다."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 없습니다. .env에 OPENAI_API_KEY=...를 설정하거나 "
            "환경 변수를 설정하세요. API 키를 코드에 작성하지 마세요."
        )
    return OpenAI(api_key=api_key)


def text_model() -> str:
    return os.getenv("OPENAI_TEXT_MODEL", DEFAULT_TEXT_MODEL)


def image_model() -> str:
    return os.getenv("OPENAI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)


def brief_context(brief: dict[str, Any]) -> str:
    competitors = ", ".join(brief.get("competitors", [])) or "없음"
    return "\n".join(
        (
            f"업종: {brief['industry']}",
            f"타겟: {brief['target']}",
            f"키워드: {', '.join(brief['keywords'])}",
            f"톤앤매너: {brief.get('tone') or '미지정'}",
            f"경쟁사: {competitors}",
            f"추가 요청사항: {brief.get('notes') or '없음'}",
        )
    )


def call_json_chat(client: OpenAI, instruction: str, prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=text_model(),
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
    )
    content = response.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("텍스트 API가 JSON이 아닌 응답을 반환했습니다.") from error
    if not isinstance(data, dict):
        raise RuntimeError("텍스트 API 응답의 최상위 값이 JSON 객체가 아닙니다.")
    return data


def require_items(data: dict[str, Any], key: str, count: int, fields: tuple[str, ...]) -> list[dict[str, str]]:
    items = data.get(key)
    if not isinstance(items, list) or len(items) != count:
        raise RuntimeError(f"API 응답의 {key}는 정확히 {count}개여야 합니다.")
    valid_items: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(field), str) or not item[field].strip() for field in fields
        ):
            raise RuntimeError(f"API 응답의 {key} 항목 형식이 올바르지 않습니다.")
        valid_items.append({field: item[field].strip() for field in fields})
    return valid_items


def generate_naming(client: OpenAI, brief: dict[str, Any]) -> list[dict[str, str]]:
    prompt = f"""다음 브랜드 브리프를 바탕으로 브랜드명 후보를 만드세요.
{brief_context(brief)}

발음하기 쉽고 경쟁사와 혼동되지 않는 후보를 정확히 4개 제안하세요. 각 후보에는
의미 또는 유래를 1~2문장으로 설명하세요. 아래 JSON 객체만 반환하세요.
{{"naming_candidates": [{{"name": "후보명", "meaning": "의미/유래"}}]}}"""
    data = call_json_chat(client, "당신은 한국 시장의 브랜드 네이밍 전문가입니다.", prompt)
    return require_items(data, "naming_candidates", 4, ("name", "meaning"))


def generate_slogans(client: OpenAI, brief: dict[str, Any], name: str, meaning: str) -> list[dict[str, str]]:
    prompt = f"""다음 브랜드의 슬로건/태그라인을 생성하세요.
{brief_context(brief)}
브랜드명: {name}
브랜드명 의미: {meaning}

브랜드의 톤앤매너에 맞는 짧고 기억하기 쉬운 문구를 정확히 3개 제안하세요.
각 문구의 의도를 한 문장으로 적으세요. 아래 JSON 객체만 반환하세요.
{{"slogans": [{{"text": "슬로건", "reason": "의도"}}]}}"""
    data = call_json_chat(client, "당신은 한국어 광고 카피라이터입니다.", prompt)
    return require_items(data, "slogans", 3, ("text", "reason"))


def limit_story(story: str) -> str:
    if len(story) <= STORY_MAX_CHARS:
        return story
    truncated = story[:STORY_MAX_CHARS]
    last_end = max(truncated.rfind(mark) for mark in ".!?")
    return truncated[: last_end + 1] if last_end >= 80 else truncated.rstrip()


def generate_story(client: OpenAI, brief: dict[str, Any], name: str, meaning: str, slogan: str) -> str:
    prompt = f"""다음 브랜드의 스토리를 한국어로 작성하세요.
{brief_context(brief)}
브랜드명: {name}
브랜드명 의미: {meaning}
슬로건: {slogan}

탄생 배경, 철학, 비전을 모두 포함해 공백 포함 250~300자로 작성하세요.
아래 JSON 객체만 반환하세요.
{{"story": "브랜드 스토리"}}"""
    data = call_json_chat(client, "당신은 브랜드 스토리텔러입니다.", prompt)
    story = data.get("story")
    if not isinstance(story, str) or not story.strip():
        raise RuntimeError("API 응답에 story 문자열이 없습니다.")
    return limit_story(story.strip())


def normalize_color(color: Any) -> dict[str, str]:
    if not isinstance(color, dict):
        raise RuntimeError("컬러 항목 형식이 올바르지 않습니다.")
    name, hex_value, reason = color.get("name"), color.get("hex"), color.get("reason")
    if not all(isinstance(value, str) and value.strip() for value in (name, hex_value, reason)):
        raise RuntimeError("컬러 항목에는 name, hex, reason이 필요합니다.")
    if not HEX_PATTERN.fullmatch(hex_value):
        raise RuntimeError(f"올바르지 않은 HEX 코드입니다: {hex_value}")
    return {"name": name.strip(), "hex": hex_value.upper(), "reason": reason.strip()}


def generate_colors(client: OpenAI, brief: dict[str, Any], name: str, slogan: str, story: str) -> dict[str, Any]:
    prompt = f"""다음 브랜드에 어울리는 컬러 팔레트를 추천하세요.
{brief_context(brief)}
브랜드명: {name}
슬로건: {slogan}
스토리: {story}

메인 컬러 1개와 서브 컬러 정확히 3개를 추천하세요. 모든 HEX는 #RRGGBB 형식이며,
각 컬러에는 이름과 선택 이유가 있어야 합니다. 아래 JSON 객체만 반환하세요.
{{"colors": {{"main_color": {{"name": "색상명", "hex": "#000000", "reason": "이유"}}, "sub_colors": [{{"name": "색상명", "hex": "#FFFFFF", "reason": "이유"}}]}}}}"""
    data = call_json_chat(client, "당신은 브랜드 컬러 전략가입니다.", prompt)
    colors = data.get("colors")
    if not isinstance(colors, dict) or not isinstance(colors.get("sub_colors"), list):
        raise RuntimeError("API 응답에 colors 구조가 없습니다.")
    sub_colors = colors["sub_colors"]
    if len(sub_colors) != 3:
        raise RuntimeError("서브 컬러는 정확히 3개여야 합니다.")
    return {
        "main_color": normalize_color(colors.get("main_color")),
        "sub_colors": [normalize_color(color) for color in sub_colors],
    }


def readable_text_color(hex_color: str) -> str:
    red, green, blue = (int(hex_color[index:index + 2], 16) for index in (1, 3, 5))
    return "#222222" if (0.299 * red + 0.587 * green + 0.114 * blue) > 160 else "#FFFFFF"


def setup_korean_font() -> None:
    """설치된 한글 폰트가 있으면 팔레트 이미지에 사용한다."""
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Malgun Gothic", "NanumGothic", "Noto Sans KR", "AppleGothic"):
        if candidate in installed:
            matplotlib.rcParams["font.family"] = candidate
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def save_color_palette(colors: dict[str, Any], output_path: Path) -> None:
    setup_korean_font()
    palette = [colors["main_color"], *colors["sub_colors"]]
    figure, axes = plt.subplots(1, len(palette), figsize=(3.1 * len(palette), 4.2))
    figure.suptitle("Brand Color Palette", fontsize=16, fontweight="bold")
    for index, (axis, color) in enumerate(zip(axes, palette)):
        axis.set_facecolor(color["hex"])
        text_color = readable_text_color(color["hex"])
        axis.text(0.5, 0.62, "MAIN" if index == 0 else f"SUB {index}", ha="center", color=text_color, fontweight="bold", transform=axis.transAxes)
        axis.text(0.5, 0.47, color["hex"], ha="center", color=text_color, fontsize=13, fontweight="bold", transform=axis.transAxes)
        axis.text(0.5, 0.20, color["name"], ha="center", va="center", color=text_color, fontsize=10, wrap=True, transform=axis.transAxes)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def generate_logos(client: OpenAI, brief: dict[str, Any], name: str, slogan: str, colors: dict[str, Any], output_dir: Path) -> list[str]:
    palette = ", ".join(color["hex"] for color in [colors["main_color"], *colors["sub_colors"]])
    files: list[str] = []
    for number, direction in enumerate(("minimal geometric symbol", "refined wordmark and emblem"), start=1):
        prompt = f"""Professional logo for the brand '{name}'.
Industry: {brief['industry']}. Target: {brief['target']}. Tone: {brief.get('tone') or 'professional'}.
Slogan concept: {slogan}. Approved colors: {palette}.
Direction: {direction}. Clean flat vector-like identity, centered on a plain light background,
strong silhouette, no product mockup, no photograph, no extra text, and no slogan text."""
        response = client.images.generate(model=image_model(), prompt=prompt, size="1024x1024")
        encoded = response.data[0].b64_json
        if not encoded:
            raise RuntimeError("이미지 API 응답에 PNG 데이터가 없습니다.")
        filename = f"logo_{number:02d}.png"
        (output_dir / filename).write_bytes(base64.b64decode(encoded))
        files.append(filename)
    return files


def api_error_message(error: Exception) -> str:
    if isinstance(error, AuthenticationError) or "api key" in str(error).lower() or "authentication" in str(error).lower():
        return "API 키가 없거나 올바르지 않습니다. .env의 OPENAI_API_KEY를 확인하고 필요하면 재발급하세요."
    return str(error)


def ask_brief_path() -> Path:
    raw = input("브리프 JSON 파일 경로를 입력하세요 (필수): ").strip().strip('"')
    if not raw:
        raise ValueError("브리프 파일 경로는 필수 입력입니다.")
    return Path(raw)


def ask_output_dir() -> Path:
    raw = input("출력 폴더 경로를 입력하세요 (기본값: ./output): ").strip().strip('"')
    return Path(raw or "./output")


def main() -> None:
    print("\nAI 브랜드 아이덴티티 생성기")
    try:
        brief_path = ask_brief_path()
        output_dir = ask_output_dir()
        brief = load_json(brief_path)
        validate_brief(brief)
    except (OSError, ValueError) as error:
        print(f"브리프 입력 오류: {error}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "brief": brief, "naming_candidates": [], "slogans": [], "story": "", "colors": None,
        "logo_files": [], "errors": {},
    }
    try:
        client = get_client()
    except Exception as error:
        result["errors"]["api_key"] = api_error_message(error)
        save_json(result, output_dir / "brand_result.json")
        print(result["errors"]["api_key"])
        print(f"빈 결과 파일을 저장했습니다: {output_dir / 'brand_result.json'}")
        return

    name, meaning, slogan = brief["industry"], "", ""
    try:
        print("[1/5] 브랜드 네이밍 생성 중...")
        result["naming_candidates"] = generate_naming(client, brief)
        name, meaning = result["naming_candidates"][0]["name"], result["naming_candidates"][0]["meaning"]
        result["brand_name"], result["brand_meaning"] = name, meaning
    except Exception as error:
        result["errors"]["naming"] = api_error_message(error)
        print(f"  네이밍 생성 실패: {result['errors']['naming']}")

    try:
        print("[2/5] 슬로건 생성 중...")
        result["slogans"] = generate_slogans(client, brief, name, meaning)
        slogan = result["slogans"][0]["text"]
        result["slogan"] = slogan
    except Exception as error:
        result["errors"]["slogan"] = api_error_message(error)
        print(f"  슬로건 생성 실패: {result['errors']['slogan']}")

    try:
        print("[3/5] 브랜드 스토리 생성 중...")
        result["story"] = generate_story(client, brief, name, meaning, slogan)
    except Exception as error:
        result["errors"]["story"] = api_error_message(error)
        print(f"  브랜드 스토리 생성 실패: {result['errors']['story']}")

    try:
        print("[4/5] 컬러 팔레트 생성 중...")
        result["colors"] = generate_colors(client, brief, name, slogan, result["story"])
        save_color_palette(result["colors"], output_dir / "color_palette.png")
    except Exception as error:
        result["errors"]["colors"] = api_error_message(error)
        print(f"  컬러 팔레트 생성 실패: {result['errors']['colors']}")

    if result["colors"]:
        try:
            print("[5/5] 로고 시안 생성 중...")
            result["logo_files"] = generate_logos(client, brief, name, slogan, result["colors"], output_dir)
        except Exception as error:
            result["errors"]["logos"] = api_error_message(error)
            print(f"  로고 시안 생성 실패: {result['errors']['logos']}")
    else:
        result["errors"]["logos"] = "컬러 팔레트 생성 실패로 로고 생성을 건너뛰었습니다."
        print(f"  로고 시안 생성 건너뜀: {result['errors']['logos']}")

    save_json(result, output_dir / "brand_result.json")
    print(f"\n완료: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
