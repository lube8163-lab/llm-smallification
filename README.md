# LLM Smallification

Gemma 4 12BをCore MLへ分割・量子化し、iPhone上で完全オフラインの
テキスト／画像／音声チャットを動かす実験リポジトリです。変換スクリプト、
SwiftUI実機プローブ、計測・品質評価ツール、実験記録を公開しています。

モデル本体や端末依存の`.mlmodelc`はGitへ含めず、配布可能な`.mlpackage`を
[Hugging Face](https://huggingface.co/lube8163/gemma-4-12b-coreml-iphone-practical-chat)
で公開しています。

## 現在の到達点

- 48層のpal4 decoder、Seq320 prefill、KV cache 512による長文生成
- Gemma 4公式MTP drafterを使ったgreedy投機デコード
- 256-token画像embedderと32-token音声embedderを同じKV生成経路へ統合
- iPhone世代別のfull-attention実行先と、融合モデルの実行先を自動選択
- 実機ログの速度・メモリ・投機受理率の機械検証

代表的なwarm runは次の通りです。短い質問、初回コンパイル、プロンプト、
MTP受理率で値は大きく変わるため、単一の数字を普遍的な速度とは扱っていません。

| 実機・条件 | 結果 |
|---|---:|
| iPhone 14、8-token高受理プローブ | 通常4.93秒/token → 投機2.06秒/token（2.40倍） |
| iPhone 14、24-token低受理プローブ | 4.20秒/token → 4.12秒/token（1.9%短縮） |
| iPhone 17、24-token、従来48分割 | 43.16秒、0.556 token/s |
| iPhone 17、先頭6層融合 | 37.56秒、0.639 token/s（13.0%短縮） |

すべての上記A/Bでtargetのgreedy token IDは一致しました。品質診断では短答・JSON・
簡単なSwift・画像・音声が機能する一方、算術誤答、冗長な前置き、特殊token流出など、
元モデル／変換後モデル／ランタイムを分離して追うべき課題も確認しています。

詳細:

- [速度改善の調査記録](docs/decode-speedup-research.md)
- [iPhone 17の速度逆転と6層融合](docs/iphone17-speed-root-cause-2026-07-24.md)
- [iPhone 14の品質診断](docs/model-quality-iphone14-2026-07-23.md)
- [iPhone 14 / 17の品質比較](docs/model-quality-iphone17-comparison-2026-07-24.md)
- [実用チャット化の記事](docs/articles/gemma4-12b-coreml-iphone-practical-chat.md)
- [投機デコードとA19最適化の続編](docs/articles/gemma4-12b-coreml-iphone-speculative-decoding.md)

## ディレクトリ

```text
docs/                 実験結果、設計メモ、記事原稿
ios/CoreMLProbe/      SwiftUI実機プローブ
scripts/              変換、配置、実機自動試験、ログ解析
patches/              外部プロジェクト向け差分
```

`models/`、`runpod-artifacts/`、実機ログ、Xcode生成物、`.mlpackage`、
`.mlmodelc`はローカル成果物としてGit管理外です。

LLaDA-MoEのdLLM/GGUF実験は
[`lube8163-lab/llada-iphone-dllm`](https://github.com/lube8163-lab/llada-iphone-dllm)
へ分離しています。

## モデルを取得する

実用チャット用の基本モデルだけを取得・検証・コンパイル・配置します。

```bash
./scripts/download_hf_gemma4_coreml_models.sh
```

投機デコードと実測採用した先頭6層融合モデルも加える場合:

```bash
HF_MODEL_PROFILE=speculative \
  ./scripts/download_hf_gemma4_coreml_models.sh
```

ダウンロードだけ行う場合は`DOWNLOAD_ONLY=1`、既存コンパイルを作り直す場合は
`FORCE=1`を指定します。Xcode、Apple signing team、十分なMac／iPhoneストレージが
別途必要です。配布構成とライセンスは
[docs/model-distribution.md](docs/model-distribution.md)を参照してください。

## 変換

主な変換スクリプト:

- `scripts/runpod_convert_gemma4_coreml_kv.py`: 単層prefill/decode/verify
- `scripts/runpod_convert_drafter_coreml.py`: MTP drafter
- `scripts/runpod_convert_gemma4_coreml_kv_fused.py`: 6層融合prefill/verify

変換はCUDA GPU環境、コンパイルと実機実行はmacOS/Xcode環境で行います。実測採用した
融合範囲は層00...05だけです。層06...11を加えると同じiPhone 17試験で5.4%遅く
なったため、配布対象にしていません。

## iPhoneへビルド・実行

モデル配置後、署名teamを環境変数で指定できます。

```bash
COREML_PROBE_DEVELOPMENT_TEAM=YOUR_TEAM_ID \
  ./scripts/build_coreml_probe_ios.sh 'generic/platform=iOS'
```

自動試験例:

```bash
COREML_PROBE_DEVICE=DEVICE_UDID \
COREML_PROBE_CHAT_PROMPT='Swiftで整数nの二乗を返す関数を書いてください。' \
COREML_PROBE_GENERATE_TOKENS=24 \
  ./scripts/run_coreml_probe_device_automation.sh \
  --kind chat \
  --require-speculative
```

主なA/B環境変数:

- `COREML_PROBE_DISABLE_SPECULATIVE=1`: target-only KV decode
- `COREML_PROBE_SPECULATIVE_TREE=0`: linear verifyだけを使用
- `COREML_PROBE_KV_FULL_ANE=0|1`: full-attention層の実行先
- `COREML_PROBE_DRAFTER_COMPUTE=cpuAndGPU|all`
- `COREML_PROBE_FUSED_COMPUTE=cpuAndGPU|all`
- `COREML_PROBE_RETAIN_VERIFY=N`: verify常駐数（診断用、既定0）

ローカルAPIは既定で無効です。信頼できるLAN内でDebugタブから明示的に有効化し、
セッションごとのBearer tokenを使います。詳細は
[docs/local-api.md](docs/local-api.md)と[SECURITY.md](SECURITY.md)にあります。

## 品質・ログ検証

```bash
python3 scripts/bench_coreml_probe_quality.py --profile full

python3 scripts/analyze_coreml_probe_log.py coremlprobe.log \
  --require-norm-lm-head \
  --require-speculative \
  --expect-layers 48 \
  --min-generated-tokens 8
```

品質スイートは小規模な診断セットで、MMLU等の統計的ベンチマークではありません。
端末間比較では同一プロンプト・greedy設定・token上限を揃え、回答文字列だけでなく
出力token IDも保存します。

## License

リポジトリ内のコードとドキュメントはMIT Licenseです。Gemma 4由来の変換済みモデル、
tokenizer、第三者プロジェクトは各上流ライセンスと利用条件に従います。
