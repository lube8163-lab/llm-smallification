# Next chat handoff: Gemma 4 12B Core ML iPhone chat

Use this prompt to continue in a fresh Codex chat.

```text
このリポジトリ `/Users/tasuku/Documents/llm小型化` の続きから作業してください。

現在の主目的は、Gemma 4 12B Unifiedの text/image/audio 対応という性質を保ちつつ、「一応はマルチモーダル入力に対応したモデル経路がメモリの少ないiPhoneでも動いた」と言える実機smokeを作ることです。速度改善はいったん優先度を下げ、既存の安定text経路 endpoint `CPU` / decoder `All` / 48 layers / Seq64 / 4-layer chunks / Cache `Run end` / Retain 0 を土台として維持します。

重要な前提:

- Repo: `https://github.com/lube8163-lab/llm-smallification`
- iOS app: `ios/CoreMLProbe/CoreMLProbe.xcodeproj`
- Current branch: `main`
- `ios/CoreMLProbe/CoreMLProbe/Models` には大きな `.mlmodelc` がローカルにあり、gitignoreされています。
- `ios/CoreMLProbe/CoreMLProbe.xcodeproj/project.pbxproj` にはXcode由来の署名設定差分が残っていることがあります。ユーザーが明示しない限り、署名差分を戻さないでください。
- 直近でアプリアイコン未反映を修正しました。原因は `ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon;` がtarget build settingsに無かったことです。Debug/Release両方に追加済みです。
- 実機向けビルドで `actool ... --app-icon AppIcon`、`CFBundleIconName = AppIcon`、`AppIcon60x60@2x.png` 生成を確認済みです。
- CoreMLProbe は `Chat` / `Probe` タブ構成になりました。
- Chat は `generate-token-loop` を使い、既定は `Cache = Run end`, `Layers = First 48`, `Max Tokens = 8` です。
- 生成 token 上限は `64` まで拡張済みです。
- endpoint compute（embedding / LM head）と decoder compute を別々に選べます。
- runner は `Token n total` と `Peak memory` をログに出します。
- runner は `Top logits token n` と、4枠が同一IDになった場合の
  `Repeated input window n` もログに出します。
- アプリは `gemma4_12b_norm_lm_head_1tok_int4_block32.mlmodelc` を優先して読みます。
  これは final RMSNorm + tied lm_head + final logit softcap を含む新endpointです。
  未配置なら旧 `gemma4_12b_lm_head_1tok_int4_block32.mlmodelc` にfallbackし、
  ログに `LM head fallback` を出します。新endpointを読めた場合は `LM head target`
  が出ます。
- `scripts/runpod_convert_gemma4_coreml_endpoints.py` を追加済みです。RunPod側で
  `--target norm-lm-head` を実行して新endpoint `.mlpackage` を作ります。
- `scripts/runpod_refresh_gemma4_coreml_endpoint.sh` はRunPod側の推奨入口です。
  実行前に `scripts/runpod_preflight_gemma4_coreml_endpoint.sh` でモデルパス、
  依存、CUDA、出力先、空き容量を確認してから変換します。
- `scripts/setup_runpod_coreml_endpoint_env.sh` はRunPod側のPython依存を入れる補助です。
  既存のCUDA対応torchを壊さないよう、既定ではtorchを再インストールしません。
- `scripts/runpod_convert_gemma4_coreml_multimodal.py` を追加済みです。RunPod側で
  32-patch image embedder `.mlpackage` を作るために使いました。
- `scripts/gemma4_token_helper.py` でホスト側 tokenizer による prompt IDs と decode ができます。
  依存は `transformers sentencepiece jinja2` です。helperの既定はraw prompt tokenizationです。
- `scripts/analyze_coreml_probe_log.py` でXcodeログを解析できます。新endpoint必須、fallback禁止、同一token検出、layer数、生成token数、peak memory上限をチェックできます。
- `scripts/preflight_coreml_probe_device.sh` は実機前のstrict gateです。asset検証、新norm+lm_head必須確認、generic iOS build、推奨実機設定の表示をまとめて行います。multimodal smoke前は `--require-image-embedder` / `--require-audio-embedder` も使えます。
- `scripts/run_coreml_probe_device_automation.sh` で実機build/install/launch/log取得/analyzeを自動化できます。接続中のiPhoneを `devicectl` で検出し、`COREML_PROBE_AUTORUN` と `COREML_PROBE_AUTO_EXIT=1` 付きで起動し、`devicectl --console` ログを保存します。`--kind image` は実image embedder smoke、`--kind audio` は実audio embedder smoke、`--kind multimodal-smoke` は合成hidden seam smokeです。
- `ios/CoreMLProbe/scripts/run_coreml_probe_device_automation.sh` はroot scriptへの薄いwrapperです。Xcode projectディレクトリからでも同じコマンドを実行できます。
- `scripts/import_coreml_probe_endpoint.sh <endpoint-mlpackage-dir-or-tar.gz>` はMac側の推奨one-shotです。RunPod成果物を受け取り、endpoint refreshとstrict preflightをまとめて実行します。
- Chat本文は bundled tokenizer でSeq64のchat token IDsへ変換されます。
  短いpromptはleft-padされます。本文に `input_ids_last64=...` /
  `input_ids_last4=...` / `input_ids=...` / `#123 #456 ...` /
  素のID列を貼った場合は、そのID列を優先してモデル入力として採用します。

実機検証の現状:

- Device: iPhone18,3 / iOS 26.4.2
- Best compute split so far: endpoint `CPU`, decoder `All`
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
- corrected norm+lm_head endpoint + bundled tokenizer + Seq64 で、endpoint `CPU`,
  decoder `All`, `Tokens = 8` が完走済み
  - peak: 666.6 MB
  - token 1: 31.0 sec
  - tokens 2-8: 平均 12.8 sec/token
  - generated: `#98275,#85896,#237354,#107,#237328,#226391,#120431,#95616`
  - `LM head target` と `Load gemma4_12b_norm_lm_head_1tok_int4_block32`
    を確認済み。`LM head fallback` は出ていません。
- `Run Stability Sweep`, endpoint `CPU`, decoder `All`, `Layers = First 48`,
  `Cache = Run end`, Retain 0, Seq64 で `Tokens = 8 -> 16` が連続完走済み
  - Tokens 8: warm avg 12.92 sec/token, peak 611.3 MB
  - Tokens 16: warm avg 13.19 sec/token, peak 605.2 MB
  - 16 tokens generated:
    `#85141,#236924,#52119,#12553,#237221,#66447,#237669,#237230,#237214,#3652,#236924,#108,#99844,#70622,#237000,#106241`
  - preferred norm+lm_head endpoint使用、fallbackなし、48 layers選択、repeated-window collapseなし。
- 追加の安定性再確認でも同じ endpoint `CPU`, decoder `All`, `Cache = Run end`,
  Retain 0, Seq64 が完走しました。
  - Tokens 8: warm avg 11.57 sec/token, peak 614.7 MB
  - Tokens 16: warm avg 12.53 sec/token, peak 611.4 MB
  - analyzerのtiming breakdownでは、warm tokenあたり decoder bundle load が
    約8.2-8.5 sec、decoder predict が約0.9-1.2 sec、LM head が約2.2-2.6 sec。
    つまり現状は推論演算よりも分割bundleのload/releaseが主な速度ネックです。
- `Run Retain Sweep`, endpoint `CPU`, decoder `All`, `Cache = Run end`,
  Retain 0 vs Retain 1, Seq64 も完走しました。
  - Retain 0: warm avg 12.18 sec/token, peak 609.1 MB
  - Retain 1: warm avg 12.32 sec/token, peak 1101.9 MB
  - 速度改善がなくメモリだけ大きく増えたため、Retain 0を既定維持してください。
- 実機自動化も確認済みです。
  - `./scripts/run_coreml_probe_device_automation.sh --skip-build --kind probe --tokens 1 --min-generated-tokens 1`
    は実機install後に自動起動、自動Probe開始、自動終了 exit 0、analyzer 0 errorで完走。peak 587.7 MB、generated `#85141`。
  - `./scripts/run_coreml_probe_device_automation.sh --skip-build --skip-install --kind chat --tokens 8`
    はChat経路でbundled tokenizerを使い、自動終了 exit 0、analyzer 0 errorで完走。warm avg 13.40 sec/token、peak 620.8 MB、generated `#85141,#236924,#238797,#237669,#227246,#236924,#145728,#237328`。
  - `./scripts/run_coreml_probe_device_automation.sh --skip-build --skip-install --kind tokens32 --timeout 1800`
    も完走。自動終了 exit 0、analyzer 0 error、warm avg 12.74 sec/token、peak 587.0 MB、32 tokens generated:
    `#85141,#236924,#52119,#12553,#237221,#66447,#237669,#237230,#237214,#3652,#236924,#108,#99844,#70622,#237000,#106241,#35071,#236951,#30886,#236945,#69144,#237512,#201132,#236951,#159734,#56980,#10965,#236951,#125561,#203956,#239542,#139281`。
  - Chat/Probe Status欄にrun statisticsを追加した後、更新済みappをinstallして `--kind chat --tokens 8` を再実行済み。自動終了 exit 0、analyzer 0 error、warm avg 12.50 sec/token、peak 621.6 MB。
- 2026-07-04の改善ループ:
  - 現行chunk4構成のbaseline: endpoint `CPU`, decoder `All`, `--kind chat --tokens 8` が完走。warm avg 13.07 sec/token、peak 617.3 MB、generated `#85141,#236924,#238797,#237669,#227246,#236924,#145728,#237328`。
  - chunk8構成（6個の8-layer decoder bundle）はasset verification自体は通るが、アプリ側guardで `unsafeComputeConfiguration` として停止。理由は8-layer Seq64 bundleがiPhone execution-plan compilationで失敗済みのため。検証後、Models配下と実機アプリはchunk4に戻して再確認済み。
  - chunk4へ戻した後の再確認: `--kind chat --tokens 8` が完走。warm avg 13.02 sec/token、peak 622.9 MB、generatedは同じ8-token列。
  - speed sweep: decoder `CPU+GPU` warm 13.95 sec/token / peak 590.1 MB、decoder `All` warm 13.85 sec/token / peak 583.1 MB、decoder `CPU+ANE` warm 51.79 sec/token / peak 209.6 MB。速度既定はdecoder `All`維持。`CPU+ANE`は低メモリだが速度面では不採用。
- 2026-07-04 RunPod / multimodal conversion:
  - RunPodのA100 SXM 80GB podを使用。画面上 `$1.49/hr`。GPUは `NVIDIA A100-SXM4-80GB`。
  - `gemma12b~~` storage上の `/workspace/gemma12b` に venv、23GBモデル、変換ログ、image/audio embedder packageがあります。
  - RunPod側model path: `/workspace/gemma12b/models/gemma-4-12b-it-qat-q4_0-unquantized`
  - 生成済みpackage: `/workspace/gemma12b/coreml-multimodal-int4/gemma4_12b_image_embedder_patches32_int4_block32.mlpackage`
  - 生成済みaudio package: `/workspace/gemma12b/coreml-multimodal-int4/gemma4_12b_audio_embedder_tokens32_int4_block32.mlpackage`
  - ローカルcompiled bundle: `ios/CoreMLProbe/CoreMLProbe/Models/gemma4_12b_image_embedder_patches32_int4_block32.mlmodelc`
  - ローカルaudio compiled bundle: `ios/CoreMLProbe/CoreMLProbe/Models/gemma4_12b_audio_embedder_tokens32_int4_block32.mlmodelc`
  - Core ML contract: `pixel_values [1,32,6912]`, `image_position_ids [1,32,2]`, `image_hidden [1,32,3840]`。
  - Audio Core ML contract: `input_features [1,32,640]`, `audio_hidden [1,32,3840]`。
- 2026-07-04 multimodal smoke:
  - `multimodal-smoke` は合成hidden `[1,64,3840]` のdecoder入力seam regressionとして残っています。実機でpeak `584.0 MB`、generated `#236812`、Log: `.derived-data/CoreMLProbe/DeviceAutomation/20260704-131948/device-console.log`
  - 本命は新しい `image-smoke` / automation `--kind image` です。実画像processor全体ではなく、32 raw patchesを実Gemma4 image embedder Core MLに通して `image_hidden` を作り、Seq64のpositions `32...63` へ入れます。
  - Command: `SEQ_LEN=64 ./scripts/run_coreml_probe_device_automation.sh --skip-build --kind image --timeout 1800`
  - Analyzer 0 errors、image embedder load `0.2879 sec`、predict `0.0161 sec`、peak `581.8 MB`、token total `34.4571 sec`、generated `#258883`、LM head fallbackなし。
  - Log: `.derived-data/CoreMLProbe/DeviceAutomation/20260704-141011/device-console.log`
  - `audio-smoke` / automation `--kind audio` も追加済みです。32 synthetic audio feature tokensを実Gemma4 audio embedder Core MLに通して `audio_hidden` を作り、Seq64のpositions `32...63` へ入れます。
  - Command: `SEQ_LEN=64 ./scripts/run_coreml_probe_device_automation.sh --kind audio --timeout 1800`
  - Analyzer 0 errors、audio embedder load `0.0389 sec`、predict `0.0027 sec`、peak `581.9 MB`、token total `33.1126 sec`、generated `#236770`、LM head fallbackなし。
  - Log: `.derived-data/CoreMLProbe/DeviceAutomation/20260704-142904/device-console.log`
  - 追加後のtext回帰も `--kind probe --tokens 8 --skip-build --skip-install` で完走。warm avg `13.29 sec/token`、peak `587.7 MB`、generated `#85141,#236924,#52119,#12553,#237221,#66447,#237669,#237230`。
  - Log: `.derived-data/CoreMLProbe/DeviceAutomation/20260704-143330/device-console.log`
- 旧linear-only LM head時点の `Tokens = 16` 再テストでは1回目が `2,123,4567,106` から開始し、
  token 6 時点で `253027,253027,253027,253027` に飽和しました。2回目は
  その飽和窓から開始したため、top logits も生成 token も同じになりました。
  peak は約 `311-320 MB` で、初回以降は約 `13.5-14.0 sec/token` です。
- 旧linear-only LM head時点の tokenizer-derived window `237234,7604,31600,3335` の実機テストでは、入力窓は
  正しく `Prompt IDs 1` に反映されました。ただし token 5 で
  `253027,253027,253027,253027` に飽和しました。この後にnorm+lm_head endpointへ更新済みです。
- endpoint `CPU+GPU`, decoder `CPU+GPU` は1token完走したが peak `1152.0 MB` で遅めでした。
- endpoint `All` は endpoint model load 中に `EXC_RESOURCE (RESOURCE_TYPE_MEMORY)` / high-water `3376 MB` でクラッシュしました。アプリ側では endpoint の ANE 系（`All`, `CPU+ANE`）を通常Pickerから外し、指定されてもロード前に失敗させるガードを追加済みです。

現在の結論:

- メモリ方針は妥当です。
- `run-end-only` が現時点の生成用cache policyとして最良です。
- endpoint は `CPU` 固定、decoder は `All` が現時点の本命です。
- Chat の既定 decoder は `All` に変更済みです。
- Retain Decoders は `0` を維持してください。Retain 1はメモリ増に対して速度改善が薄く、2以上はUI上限から外しています。
- Chat UX改善として、Chatの既定 `Max Tokens = 8`、8/16/32 segmented preset、
  生成IDのdecode表示、EOS/turn delimiter停止を実装済みです。
- Probeに `Run Retain Sweep` を追加済みです。endpoint `CPU`, decoder `All`,
  `Tokens = 8`, Retain 0 vs Retain 1 を比較できます。実機結果ではRetain 1は不採用です。
- 起動時自動テストと完了後の自動アプリ停止を実装済みです。`COREML_PROBE_AUTORUN` / `--autorun=` は `probe`, `chat`, `stability-sweep`, `retention-sweep`, `speed-sweep` を受け付け、`COREML_PROBE_AUTO_EXIT=1` / `--auto-exit` で終了します。
- Chat/ProbeのStatus欄は、run完了後にgenerated/target tokens、warm avg、all-token avg、peak memory、停止理由、decoder load / decoder predict / LM head のper-token timingを表示します。
- `Tokens = 16` と `Tokens = 32` は安定していますが遅いです。通常Chat既定は8のままにしてください。
- 8-layer decoder chunk化は速度改善候補から外してください。Seq64では実機側のexecution-plan compilation失敗によりアプリが明示的にブロックします。
- RunPodは次のローカルapp-side作業には不要です。2026-07-04時点でA100 SXM pod上にモデル/venv/image/audio embedder成果物があるので、fuller image/audio processor変換をすぐ続けるならstorageは残してください。変換しない間はpod停止/terminate候補です。

マルチモーダル方針:

- Gemma 4 12B Unified自体は text/image/audio input 対応です。
- 現在のCore ML bundleは text embedding / 48 decoder layers / norm+lm_head に加えて、32-patch image embedderと32-token audio embedderを含みます。
- `multimodal-smoke` でdecoder入力seamに `[1,64,3840]` hidden を渡すSwift/Core ML plumbingは実機確認済みです。
- `image-smoke` で実Gemma4 image embedder Core ML endpoint -> 48-layer decoder -> norm+lm_head が実機確認済みです。
- `audio-smoke` で実Gemma4 audio embedder Core ML endpoint -> 48-layer decoder -> norm+lm_head も実機確認済みです。
- ただしHugging Face標準processorの `<|image|>` は256 image tokens / `pixel_values [1,280,6912]` になり、現行Seq64には収まりません。現在の到達点は32 raw patches / 32 synthetic audio featuresのmicro smokeで、semantic image/audio QAではありません。
- 次に広げるなら real image/audio preprocessing、またはSeq/token budgetを増やしたfuller pathです。

次にやること:

1. まず `git status --short` を確認してください。
2. `docs/handoff-current.md` と `docs/2026-06-27-runpod-coreml-probe.md` を読んでください。
3. `SEQ_LEN=64 ./scripts/preflight_coreml_probe_device.sh --skip-build --require-image-embedder --require-audio-embedder` でasset状態を確認してください。
4. Chat既定は endpoint `CPU`, decoder `All`, `Layers = First 48`, `Cache = Run end`, Retain 0, `Max Tokens = 8` を維持してください。
5. 以後の実機確認は `scripts/run_coreml_probe_device_automation.sh` を使ってください。初回install後は `--skip-build --skip-install` を付けると起動/ログ取得だけになります。
6. 次の本命は、real image/audio preprocessingをアプリへ入れるか、fuller image/audioを目指してSeq/token budgetを再設計するかの選択です。速度目的の再分割より、今はmodality pathの忠実度を上げる作業を優先してください。
   - 再分割については8-layer chunkは不採用です。
7. 実機ログを保存したら `./scripts/analyze_coreml_probe_log.py <log> --require-norm-lm-head --fail-on-repeat --expect-layers 48 --min-generated-tokens 8 --max-peak-mb 900` を実行してください。image-smokeは1tokenなので `--min-generated-tokens 1` にします。自動化スクリプトは既定でこの解析も行います。
8. Chat本文は bundled tokenizer で入力できます。必要なら `python3 scripts/gemma4_token_helper.py --prompt "..."` の `input_ids` / `input_ids_last64` 相当を貼り付けても検証できます。生成 ID の確認は `--decode-ids` です。
9. 意図した差分だけcommit/pushしてください。大きなモデルやXcode署名差分を不用意にstageしないでください。
```
