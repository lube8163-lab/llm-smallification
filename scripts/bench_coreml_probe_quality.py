#!/usr/bin/env python3
"""Run a reproducible semantic-quality suite against CoreMLProbe's local API.

The suite intentionally mixes short, objectively scored tasks with open-ended
tasks that need human review. It records the exact decoded response, token IDs,
end-to-end generation time, and observed tokens/second after every case so a
partial run remains useful if the device disconnects.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QualityCase:
    case_id: str
    category: str
    prompt: str
    tokens: int
    required_patterns: tuple[str, ...] = ()
    review_note: str = ""
    attachment: str | None = None


TEXT_CASES = (
    QualityCase(
        case_id="common_knowledge",
        category="常識",
        prompt="日本の首都はどこですか。都市名だけを答えてください。",
        tokens=8,
        required_patterns=(r"東京",),
    ),
    QualityCase(
        case_id="arithmetic",
        category="算術",
        prompt="17×23を計算してください。答えの数字だけを書いてください。",
        tokens=8,
        required_patterns=(r"391",),
    ),
    QualityCase(
        case_id="logic",
        category="論理",
        prompt=(
            "すべての鳥は動物です。ペンギンは鳥です。"
            "したがってペンギンは動物ですか。はい・いいえの後に理由を一文で答えてください。"
        ),
        tokens=20,
        required_patterns=(r"はい", r"鳥.*動物|動物.*鳥"),
    ),
    QualityCase(
        case_id="instruction_json",
        category="指示追従",
        prompt=(
            "赤、青、緑の3語を、この順序を変えずJSON配列だけで出力してください。"
            "説明やコードフェンスは不要です。"
        ),
        tokens=16,
        required_patterns=(r"\[", r"赤.*青.*緑", r"\]"),
    ),
    QualityCase(
        case_id="coding_swift",
        category="コーディング",
        prompt=(
            "Swiftで整数nの二乗を返す関数 square(_ n: Int) -> Int を実装してください。"
            "コードだけを出力してください。"
        ),
        tokens=32,
        required_patterns=(r"func\s+square", r"Int", r"n\s*\*\s*n"),
        review_note="構文が完結し、入力2で4、入力-3で9になるか確認する。",
    ),
    QualityCase(
        case_id="abstract_reasoning",
        category="抽象的質問",
        prompt="自由と責任が両立するために必要な条件を、2点に絞って簡潔に説明してください。",
        tokens=32,
        review_note="2点が区別され、自由と責任の関係を説明し、自己矛盾がないか確認する。",
    ),
    QualityCase(
        case_id="uncertainty",
        category="幻覚耐性",
        prompt=(
            "架空の学者『森崎レオン』が提唱した量子みかん理論の発表年を答えてください。"
            "確認できない場合は推測せず、その旨を答えてください。"
        ),
        tokens=24,
        required_patterns=(r"架空|確認でき|分かりません|わかりません|不明|存在し",),
    ),
    QualityCase(
        case_id="translation",
        category="翻訳",
        prompt=(
            "次の英文を自然な日本語に翻訳してください。"
            " The experiment failed because the battery was empty."
        ),
        tokens=24,
        required_patterns=(r"実験", r"バッテリー|電池", r"空|切れ|なかった"),
    ),
)

FOLLOWUP_TEXT_CASES = (
    QualityCase(
        case_id="arithmetic_repeat",
        category="算術・再現性",
        prompt="17×23を計算してください。答えの数字だけを書いてください。",
        tokens=8,
        required_patterns=(r"391",),
        review_note="初回の誤答419がgreedy生成で再現するか確認する。",
    ),
    QualityCase(
        case_id="arithmetic_working",
        category="算術・途中計算",
        prompt="17×23を途中計算一行と最終結果一行で答えてください。",
        tokens=32,
        required_patterns=(r"391",),
    ),
    QualityCase(
        case_id="abstract_compact",
        category="抽象・短答",
        prompt=(
            "自由と責任の両立条件を2点答えてください。"
            "必ず「1:」「2:」の2行だけにし、各行12文字以内にしてください。"
        ),
        tokens=32,
        required_patterns=(r"1\s*[:：]", r"2\s*[:：]"),
        review_note="2点が意味的に異なり、自由と責任の関係に関連するか確認する。",
    ),
    QualityCase(
        case_id="uncertainty_compact",
        category="幻覚耐性・短答",
        prompt=(
            "『森崎レオンの量子みかん理論』は実在を確認できますか。"
            "「確認できる」「確認できない」のどちらかだけで答えてください。"
        ),
        tokens=8,
        required_patterns=(r"確認できない",),
    ),
    QualityCase(
        case_id="translation_direct",
        category="翻訳・短答",
        prompt=(
            "説明を付けず次の一文だけを日本語に訳してください:"
            " The experiment failed because the battery was empty."
        ),
        tokens=24,
        required_patterns=(r"実験", r"バッテリー|電池", r"空|切れ|なかった"),
    ),
)

EXTENDED_TEXT_CASES = (
    QualityCase(
        case_id="extended_arithmetic",
        category="算術・検算",
        prompt=(
            "17×23を計算してください。途中計算を示し、最後の行を"
            "「答え: 数字」の形式にしてください。"
        ),
        tokens=64,
        required_patterns=(r"391", r"答え\s*[:：]\s*391"),
        review_note="途中計算と最終結果が一致し、17×20 + 17×3 = 391等の検算が成立するか確認する。",
    ),
    QualityCase(
        case_id="extended_logic",
        category="論理・反例",
        prompt=(
            "命題「ある鳥は飛べない」と「すべてのペンギンは鳥である」から、"
            "「すべてのペンギンは飛べない」は必ず導けますか。"
            "結論を最初に書き、理由を2文以内で説明してください。"
        ),
        tokens=64,
        required_patterns=(r"導けない|必ず.*ない|いいえ", r"ある.*すべて|情報.*不足|ペンギン"),
        review_note="存在命題から全称命題を導けないことを正しく説明できるか確認する。",
    ),
    QualityCase(
        case_id="extended_instruction_json",
        category="指示追従・構造化",
        prompt=(
            "次の情報だけを有効なJSONオブジェクト1個で出力してください。"
            "nameはapple、colorsはredとgreenの配列、countは2。"
            "説明、Markdown、追加キーは禁止です。"
        ),
        tokens=64,
        required_patterns=(
            r'"name"\s*:\s*"apple"',
            r'"colors"\s*:\s*\[\s*"red"\s*,\s*"green"\s*\]',
            r'"count"\s*:\s*2',
        ),
        review_note="JSONとしてparse可能で、指定外キーや前後の説明がないか確認する。",
    ),
    QualityCase(
        case_id="extended_coding_swift",
        category="コーディング・長文",
        prompt=(
            "昇順の整数配列からtargetの位置を返し、存在しなければnilを返す"
            "Swift関数 binarySearch(_ values: [Int], target: Int) -> Int? を"
            "whileループで実装してください。コードだけを出力してください。"
        ),
        tokens=96,
        required_patterns=(
            r"func\s+binarySearch",
            r"while",
            r"return\s+nil",
            r"target",
        ),
        review_note="Swiftとしてコンパイルし、先頭・中央・末尾・不在・空配列で確認する。",
    ),
    QualityCase(
        case_id="extended_abstract",
        category="抽象的質問・長文",
        prompt=(
            "個人の自由と社会的責任を両立させる原則を2つ挙げてください。"
            "各原則について、なぜ両立に役立つかを1文で説明してください。"
        ),
        tokens=96,
        review_note="原則が2つに分かれ、理由が明示され、自己矛盾や不自然な反復がないか確認する。",
    ),
    QualityCase(
        case_id="extended_uncertainty",
        category="幻覚耐性・出典",
        prompt=(
            "架空の論文『Neural Citrus Dynamics』（森崎レオン、2019年）の"
            "DOIと主要結論を教えてください。実在を確認できない情報は作らず、"
            "確認できないと明記してください。"
        ),
        tokens=64,
        required_patterns=(r"確認でき|架空|実在しない|不明|存在しない",),
        review_note="DOIや研究内容を捏造していないか確認する。",
    ),
    QualityCase(
        case_id="extended_translation",
        category="翻訳・複文",
        prompt=(
            "次を自然な日本語に翻訳し、訳文だけを出力してください。"
            " Although the prototype was faster, the team postponed the release "
            "because its memory use was unpredictable and the safety tests were incomplete."
        ),
        tokens=64,
        required_patterns=(
            r"試作|プロトタイプ",
            r"速",
            r"延期",
            r"メモリ",
            r"安全(?:性)?.*(?:テスト|試験)",
        ),
        review_note="因果・譲歩関係とincompleteの意味が保たれ、余計な説明がないか確認する。",
    ),
)


def make_opener() -> urllib.request.OpenerDirector:
    # Link-local iPhone traffic must never be sent through a configured proxy.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    token: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    body = None
    method = "GET"
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def score_response(case: QualityCase, text: str) -> tuple[str, list[str]]:
    if not case.required_patterns:
        return "review", []
    missing = [
        pattern
        for pattern in case.required_patterns
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is None
    ]
    return ("pass" if not missing else "fail"), missing


def markdown(results: list[dict[str, Any]], status: dict[str, Any]) -> str:
    lines = [
        "# CoreMLProbe quality benchmark",
        "",
        f"- generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- device status: `{json.dumps(status, ensure_ascii=False)}`",
        "",
        "| case | category | score | tokens | seconds | tok/s | response |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        response = str(result.get("reply_text", "")).replace("\n", "<br>").replace("|", "\\|")
        lines.append(
            "| {case_id} | {category} | {score} | {token_count} | "
            "{seconds:.2f} | {tokens_per_second:.3f} | {response} |".format(
                **result,
                response=response,
            )
        )
    objective = [item for item in results if item["score"] in {"pass", "fail"}]
    passed = sum(item["score"] == "pass" for item in objective)
    total_tokens = sum(int(item["token_count"]) for item in results)
    total_seconds = sum(float(item["seconds"]) for item in results)
    lines.extend(
        [
            "",
            f"- objective checks: {passed}/{len(objective)} passed",
            f"- aggregate: {total_tokens} tokens / {total_seconds:.2f}s = "
            f"{(total_tokens / total_seconds if total_seconds else 0):.3f} tok/s",
            "",
            "## Review notes",
            "",
        ]
    )
    for result in results:
        if result.get("review_note"):
            lines.append(f"- `{result['case_id']}`: {result['review_note']}")
        if result.get("missing_patterns"):
            lines.append(
                f"- `{result['case_id']}` missing: "
                + ", ".join(f"`{pattern}`" for pattern in result["missing_patterns"])
            )
    return "\n".join(lines) + "\n"


def checkpoint(
    output_dir: Path,
    results: list[dict[str, Any]],
    status: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps({"status": status, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(markdown(results, status), encoding="utf-8")


def attachment_payload(case: QualityCase, args: argparse.Namespace) -> dict[str, Any]:
    if case.attachment == "image":
        if args.image is None:
            raise ValueError("image case requested without --image")
        return {
            "image_b64": base64.b64encode(args.image.read_bytes()).decode("ascii"),
            "image_norm": "unit",
        }
    if case.attachment == "audio":
        if args.audio is None:
            raise ValueError("audio case requested without --audio")
        return {"audio_b64": base64.b64encode(args.audio.read_bytes()).decode("ascii")}
    return {}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("COREML_PROBE_API_BASE", "http://169.254.139.200:8765"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/coreml-quality") / datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("full", "followup", "extended"),
        default="full",
        help=(
            "full runs the primary suite; followup isolates prompt/cap effects; "
            "extended uses 64-96 token caps for deeper quality review"
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="run only the named case; repeat to select multiple cases",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("COREML_PROBE_API_TOKEN", "")
    if len(token) < 16:
        parser.error("set COREML_PROBE_API_TOKEN to the active API bearer token")

    if args.profile == "full":
        cases = list(TEXT_CASES)
    elif args.profile == "followup":
        cases = list(FOLLOWUP_TEXT_CASES)
    else:
        cases = list(EXTENDED_TEXT_CASES)
    if args.image is not None:
        if args.profile == "full":
            cases.append(
                QualityCase(
                    case_id="image_ui",
                    category="画像理解",
                    prompt=(
                        "画像に表示されているクラウドGPUサービス名、Podの状態、"
                        "表示されている接続方法を簡潔に答えてください。"
                    ),
                    tokens=32,
                    required_patterns=(
                        r"RunPod",
                        r"実行|稼働|Running|Ready|接続",
                        r"SSH|Jupyter|Terminal",
                    ),
                    review_note="画面にない物体や状態を捏造していないか確認する。",
                    attachment="image",
                )
            )
        elif args.profile == "followup":
            cases.extend(
                (
                    QualityCase(
                        case_id="image_pod_state",
                        category="画像理解・状態",
                        prompt="画像のPodは動作中ですか、停止中ですか。どちらかだけ答えてください。",
                        tokens=8,
                        required_patterns=(r"動作中|稼働中|Running",),
                        attachment="image",
                    ),
                    QualityCase(
                        case_id="image_connections",
                        category="画像理解・接続",
                        prompt="画像に表示された接続方法を、名称だけ3つまで列挙してください。",
                        tokens=32,
                        required_patterns=(r"HTTP services", r"SSH", r"SSH over exposed TCP"),
                        attachment="image",
                    ),
                )
            )
        else:
            cases.append(
                QualityCase(
                    case_id="extended_image_ui",
                    category="画像理解・詳細",
                    prompt=(
                        "画像のクラウドGPU管理画面を読み取り、サービス名、Pod名、"
                        "稼働状態、接続方法、画面に見えるポート番号を箇条書きで説明してください。"
                        "読めない項目は推測しないでください。"
                    ),
                    tokens=96,
                    required_patterns=(
                        r"RunPod",
                        r"wicked_orange_lion",
                        r"Running|Ready|動作中|稼働中",
                        r"Jupyter",
                        r"SSH",
                        r"8888",
                    ),
                    review_note=(
                        "Web terminal/19123、SSH over exposed TCP/16263→22も読めるか、"
                        "画面外の情報を捏造しないか確認する。"
                    ),
                    attachment="image",
                )
            )
    if args.audio is not None:
        cases.append(
            QualityCase(
                case_id="audio_arithmetic",
                category="音声理解",
                prompt="音声内の質問に答えてください。答えの数字だけを書いてください。",
                tokens=8,
                required_patterns=(r"4|四",),
                attachment="audio",
            )
        )
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case.case_id in selected]
        missing = selected.difference(case.case_id for case in cases)
        if missing:
            parser.error("unknown case-id(s): " + ", ".join(sorted(missing)))

    opener = make_opener()
    base_url = args.base_url.rstrip("/")
    try:
        status = request_json(
            opener,
            f"{base_url}/status",
            token,
            timeout=min(args.timeout, 30),
        )
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        print(f"status request failed: {error}", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    checkpoint(args.output_dir, results, status)
    print(f"output: {args.output_dir}")
    print(f"status: {json.dumps(status, ensure_ascii=False)}")

    if not args.skip_warmup:
        warmup = {"prompt": "「はい」とだけ答えてください。", "tokens": 4}
        started = time.monotonic()
        response = request_json(
            opener,
            f"{base_url}/generate",
            token,
            payload=warmup,
            timeout=args.timeout,
        )
        print(
            f"warmup: ok={response.get('ok')} "
            f"seconds={time.monotonic() - started:.2f} reply={response.get('reply_text')!r}"
        )

    for index, case in enumerate(cases, start=1):
        payload: dict[str, Any] = {"prompt": case.prompt, "tokens": case.tokens}
        payload.update(attachment_payload(case, args))
        print(f"[{index}/{len(cases)}] {case.case_id} ({case.category})", flush=True)
        wall_started = time.monotonic()
        try:
            response = request_json(
                opener,
                f"{base_url}/generate",
                token,
                payload=payload,
                timeout=args.timeout,
            )
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            result = {
                "case_id": case.case_id,
                "category": case.category,
                "prompt": case.prompt,
                "requested_tokens": case.tokens,
                "ok": False,
                "score": "error",
                "reply_text": "",
                "reply_tokens": [],
                "token_count": 0,
                "seconds": time.monotonic() - wall_started,
                "tokens_per_second": 0.0,
                "error": str(error),
                "review_note": case.review_note,
                "missing_patterns": list(case.required_patterns),
            }
            results.append(result)
            checkpoint(args.output_dir, results, status)
            print(f"  error: {error}", file=sys.stderr)
            continue

        reply_text = str(response.get("reply_text", ""))
        reply_tokens = response.get("reply_tokens")
        if not isinstance(reply_tokens, list):
            reply_tokens = []
        seconds = float(response.get("seconds") or (time.monotonic() - wall_started))
        score, missing = score_response(case, reply_text)
        result = {
            "case_id": case.case_id,
            "category": case.category,
            "prompt": case.prompt,
            "requested_tokens": case.tokens,
            "ok": bool(response.get("ok")),
            "score": score,
            "reply_text": reply_text,
            "reply_tokens": reply_tokens,
            "token_count": len(reply_tokens),
            "seconds": seconds,
            "tokens_per_second": len(reply_tokens) / seconds if seconds else 0.0,
            "summary": response.get("summary", ""),
            "review_note": case.review_note,
            "missing_patterns": missing,
        }
        results.append(result)
        checkpoint(args.output_dir, results, status)
        print(
            f"  score={score} tokens={len(reply_tokens)} seconds={seconds:.2f} "
            f"tok/s={result['tokens_per_second']:.3f} reply={reply_text!r}",
            flush=True,
        )

    objective = [item for item in results if item["score"] in {"pass", "fail"}]
    passed = sum(item["score"] == "pass" for item in objective)
    print(f"objective checks: {passed}/{len(objective)} passed")
    print(f"summary: {args.output_dir / 'summary.md'}")
    return 0 if all(item.get("ok") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
