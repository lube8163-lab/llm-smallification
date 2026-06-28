# Next chat handoff: Gemma 4 12B Core ML iPhone chat

Use this prompt to continue in a fresh Codex chat.

```text
このリポジトリ `/Users/tasuku/Documents/llm小型化` の続きから作業してください。

目的は、Gemma 4 12B text-only Core ML chat/probe を、クラッシュしない範囲で高速化しつつ、固定4token windowの運用を少しずつ実用寄りにすることです。実用速度はまだ遅いですが、技術検証としてiPhone上で12Bを逐次ロードして動かす方針を続けます。

重要な前提:

- Repo: `https://github.com/lube8163-lab/llm-smallification`
- iOS app: `ios/CoreMLProbe/CoreMLProbe.xcodeproj`
- Current branch: `main`
- `ios/CoreMLProbe/CoreMLProbe/Models` には大きな `.mlmodelc` がローカルにあり、gitignoreされています。
- `ios/CoreMLProbe/CoreMLProbe.xcodeproj/project.pbxproj` にはXcode由来の署名設定差分が残っていることがあります。ユーザーが明示しない限り、署名差分を戻さないでください。
- 直近でアプリアイコン未反映を修正しました。原因は `ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon;` がtarget build settingsに無かったことです。Debug/Release両方に追加済みです。
- 実機向けビルドで `actool ... --app-icon AppIcon`、`CFBundleIconName = AppIcon`、`AppIcon60x60@2x.png` 生成を確認済みです。
- CoreMLProbe は `Chat` / `Probe` タブ構成になりました。
- Chat は `generate-token-loop` を使い、既定は `Cache = Run end`, `Layers = First 48`, `Tokens = 2` です。
- 生成 token 上限は `32` まで拡張済みです。
- endpoint compute（embedding / LM head）と decoder compute を別々に選べます。
- runner は `Token n total` と `Peak memory` をログに出します。
- runner は `Top logits token n` と、4枠が同一IDになった場合の
  `Repeated input window n` もログに出します。
- アプリは `gemma4_12b_norm_lm_head_1tok_int4_block32.mlmodelc` を優先して読みます。
  これは final RMSNorm + tied lm_head + final logit softcap を含む新endpointです。
  未配置なら旧 `gemma4_12b_lm_head_1tok_int4_block32.mlmodelc` にfallbackし、
  ログに `LM head fallback` を出します。
- `scripts/runpod_convert_gemma4_coreml_endpoints.py` を追加済みです。RunPod側で
  `--target norm-lm-head` を実行して新endpoint `.mlpackage` を作ります。
- `scripts/gemma4_token_helper.py` でホスト側 tokenizer による prompt -> last4 IDs と decode ができます。
  依存は `transformers sentencepiece jinja2` です。固定4token検証ではchat templateの末尾がassistant prefixに寄りやすいため、helperの既定はraw prompt tokenizationです。
- Chat本文はまだオンデバイスtokenizerを通っていません。本文に
  `input_ids_last4=...` / `input_ids=...` / `#123 #456 #789 #10` /
  素の4IDを貼った場合だけ、それをモデル入力として採用します。普通の
  自然文だけなら `Token window` 欄が入力になります。

実機検証の現状:

- Device: iPhone18,3 / iOS 26.4.2
- Best compute split so far: endpoint `CPU`, decoder `CPU+GPU`
- Layers: First 48
- Cache policy: Run end
- CPU-only `generate-token-loop`, `Tokens = 8` は完走済み
  - total timed steps: 約 459.1 sec
  - peak: 226.2 MB
  - token 1: 67.5 sec
  - tokens 2-8: 平均 54.7 sec/token
- endpoint `CPU`, decoder `CPU+GPU`, `generate-token-loop`, `Tokens = 8` も完走済み
  - token totals sum: 約 115.9 sec
  - peak: 316.2 MB
  - token 1: 20.1 sec
  - tokens 2-8: 平均 13.7 sec/token
  - 生成は現状 `#253027` に寄るため、品質ではなく速度・安定性評価として見る
- `Tokens = 16` の再テストでは1回目が `2,123,4567,106` から開始し、
  token 6 時点で `253027,253027,253027,253027` に飽和しました。2回目は
  その飽和窓から開始したため、top logits も生成 token も同じになりました。
  peak は約 `311-320 MB` で、初回以降は約 `13.5-14.0 sec/token` です。
- tokenizer-derived window `237234,7604,31600,3335` の実機テストでは、入力窓は
  正しく `Prompt IDs 1` に反映されました。ただし token 5 で
  `253027,253027,253027,253027` に飽和しました。現行LM head bundleのMILを確認すると
  linearのみでfinal RMSNorm/softcapが無いため、次はendpoint再変換が本命です。
- endpoint `CPU+GPU`, decoder `CPU+GPU` は1token完走したが peak `1152.0 MB` で遅めでした。
- endpoint `All` は endpoint model load 中に `EXC_RESOURCE (RESOURCE_TYPE_MEMORY)` / high-water `3376 MB` でクラッシュしました。アプリ側では endpoint の ANE 系（`All`, `CPU+ANE`）を通常Pickerから外し、指定されてもロード前に失敗させるガードを追加済みです。

現在の結論:

- メモリ方針は妥当です。
- `run-end-only` が現時点の生成用cache policyとして最良です。
- endpoint は `CPU` 固定、decoder は `CPU+GPU` が現時点の本命です。
- Chat の既定 decoder は `CPU+GPU` に変更済みです。
- 次は RunPodで `norm+lm_head` endpointを再変換し、実機でfallbackなしロードを確認してから
  tokenizer-derived windowで `Tokens = 8 -> 16` を再確認するのがよいです。

マルチモーダル方針:

- Gemma 4 12B Unified自体は text/image/audio input 対応です。
- ただし現在のCore ML bundleは text embedding / 48 decoder layers / LM head のtext-onlyです。
- チャットUIは画像・音声添付スロットを将来拡張用に用意してよいですが、実推論へ接続するのは後段です。
- 画像・音声を実際に扱うには、modality embedding/projection pathをCore ML化し、3840幅のdecoder入力へ流す追加作業が必要です。

次にやること:

1. まず `git status --short` を確認してください。
2. `docs/handoff-current.md` と `docs/2026-06-27-runpod-coreml-probe.md` を読んでください。
3. RunPod側で `python scripts/runpod_convert_gemma4_coreml_endpoints.py --target norm-lm-head` を実行し、新しい `gemma4_12b_norm_lm_head_1tok_int4_block32.mlpackage` を作ってください。
4. Mac側で `./scripts/refresh_coreml_probe_endpoint.sh <endpoint-mlpackage-dir>` を実行してください。これでcompile、endpoint-only copy、strict verify、generic iOS buildまで一括で走ります。
5. 実機では endpoint `CPU`, decoder `CPU+GPU`, `Layers = First 48`, `Cache = Run end` のまま、ログに `Load gemma4_12b_norm_lm_head_1tok_int4_block32` が出て `LM head fallback` が出ないことを確認してください。
6. token window は `python3 scripts/gemma4_token_helper.py --prompt "..."` の `input_ids_last4` をChat本文または `Token window` 欄に貼って、まず `Tokens = 8`、安定すれば `16` を試してください。生成 ID の確認は `--decode-ids` です。
7. 意図した差分だけcommit/pushしてください。大きなモデルやXcode署名差分を不用意にstageしないでください。
```
