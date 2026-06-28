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
- `scripts/gemma4_token_helper.py` でホスト側 tokenizer による prompt -> last4 IDs と decode ができます。

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
- 次は、実機で accelerator の安定範囲を小さい layer 数から測りつつ、host tokenizer helper で固定4token windowを改善するのがよいです。

マルチモーダル方針:

- Gemma 4 12B Unified自体は text/image/audio input 対応です。
- ただし現在のCore ML bundleは text embedding / 48 decoder layers / LM head のtext-onlyです。
- チャットUIは画像・音声添付スロットを将来拡張用に用意してよいですが、実推論へ接続するのは後段です。
- 画像・音声を実際に扱うには、modality embedding/projection pathをCore ML化し、3840幅のdecoder入力へ流す追加作業が必要です。

次にやること:

1. まず `git status --short` を確認してください。
2. `docs/handoff-current.md` と `docs/2026-06-27-runpod-coreml-probe.md` を読んでください。
3. 実機ではまず `Tokens = 1`, `Cache = Run end` のまま、decoder compute を `CPU+GPU` または `All` にして `First 1 -> 8 -> 16 -> 32 -> 48` の順に攻めてください。
4. decoder 側が安定したら endpoint compute（embedding / LM head）を `CPU`, `CPU+GPU`, `All` で比較してください。
5. token window は `python3 scripts/gemma4_token_helper.py --prompt "..."` の `input_ids_last4` を使って改善してください。生成 ID の確認は `--decode-ids` です。
6. 変更後は `./scripts/build_coreml_probe_ios.sh` と、必要なら `./scripts/build_coreml_probe_ios.sh 'generic/platform=iOS'` を実行してください。
7. 意図した差分だけcommit/pushしてください。大きなモデルやXcode署名差分を不用意にstageしないでください。
```
