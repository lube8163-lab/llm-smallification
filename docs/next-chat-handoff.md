# Next chat handoff: Gemma 4 12B Core ML iPhone chat

Use this prompt to continue in a fresh Codex chat.

```text
このリポジトリ `/Users/tasuku/Documents/llm小型化` の続きから作業してください。

目的は、Gemma 4 12B text-only Core ML probe を、まずはテキストチャットUIへ発展させることです。実用速度はまだ遅いですが、技術検証としてiPhone上で12Bを逐次ロードして動かす方針を続けます。

重要な前提:

- Repo: `https://github.com/lube8163-lab/llm-smallification`
- iOS app: `ios/CoreMLProbe/CoreMLProbe.xcodeproj`
- Current branch: `main`
- `ios/CoreMLProbe/CoreMLProbe/Models` には大きな `.mlmodelc` がローカルにあり、gitignoreされています。
- `ios/CoreMLProbe/CoreMLProbe.xcodeproj/project.pbxproj` にはXcode由来の署名設定差分が残っていることがあります。ユーザーが明示しない限り、署名差分を戻さないでください。
- 直近でアプリアイコン未反映を修正しました。原因は `ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon;` がtarget build settingsに無かったことです。Debug/Release両方に追加済みです。
- 実機向けビルドで `actool ... --app-icon AppIcon`、`CFBundleIconName = AppIcon`、`AppIcon60x60@2x.png` 生成を確認済みです。

実機検証の現状:

- Device: iPhone18,3 / iOS 26.4.2
- Compute: CPU
- Layers: First 48
- Cache policy: Run end
- `generate-token-loop`, `Tokens = 8` が完走済み
- 8token結果:
  - total timed steps: 約 459.1 sec
  - peak: 226.2 MB
  - token 1: 67.5 sec, peak 226.2 MB
  - tokens 2-8: 平均 54.7 sec/token, peak 39-43 MB付近
  - 後続tokenではdecoder loadがほぼ消え、decoder predictが主なボトルネック

現在の結論:

- メモリ方針は妥当です。
- `run-end-only` が現時点の生成用cache policyとして最良です。
- 次は速度最適化よりも、先にテキストチャット化・tokenizer/prompt formatting・固定4token windowの扱いを整えるのがよいです。

マルチモーダル方針:

- Gemma 4 12B Unified自体は text/image/audio input 対応です。
- ただし現在のCore ML bundleは text embedding / 48 decoder layers / LM head のtext-onlyです。
- チャットUIは画像・音声添付スロットを将来拡張用に用意してよいですが、実推論へ接続するのは後段です。
- 画像・音声を実際に扱うには、modality embedding/projection pathをCore ML化し、3840幅のdecoder入力へ流す追加作業が必要です。

次にやること:

1. まず `git status --short` を確認してください。
2. `docs/handoff-current.md` と `docs/2026-06-27-runpod-coreml-probe.md` を読んでください。
3. そのうえで、CoreMLProbeを「Probe画面 + Chat画面」の構成にするか、まずChat風UIを既存画面内に追加するかを判断してください。
4. 最初の実装はテキストチャット優先:
   - user message入力欄
   - assistant生成中状態
   - 生成結果token列の表示
   - mode/cache/layers/tokensの最低限設定
   - 将来の画像・音声添付ボタンは未接続またはplaceholderでよい
5. 実機メモリを壊さないため、生成時は `Cache = Run end` を既定にしてください。
6. 変更後は `./scripts/build_coreml_probe_ios.sh` と、必要なら `./scripts/build_coreml_probe_ios.sh 'generic/platform=iOS'` を実行してください。
7. 意図した差分だけcommit/pushしてください。大きなモデルやXcode署名差分を不用意にstageしないでください。
```
