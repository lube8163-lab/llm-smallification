# iPhone 17 速度逆転の原因調査と改善結果（2026-07-24）

## 結論

iPhone 17 が iPhone 14 より遅く見えた主因は、A19 の推論性能ではなく、iPhone 14 での不安定動作を避けるために入れた full-attention 8 層の CPU+GPU 強制が全端末へ適用されていたことと、投機的デコードで 48 個の verify モデルを各 sweep でロードする構成だった。

同じ iPhone 17、同じ質問、同じ 24 出力トークンで直接 A/B した結果、full-attention 層を ANE に戻すと 52.61 秒から 42.38 秒へ短縮した。最終構成ではドラフターの失敗する ANE コンパイルも避け、43.16 秒、0.556 tok/s、ピーク 665.6 MB となった。出力テキストと全 24 token ID は基準構成と完全一致した。

## 端末と試験条件

- iPhone 17: iPhone18,3、iOS 26.5.2
- iPhone 14: iPhone14,7、iOS 26.5.2
- ターゲット: Gemma 4 12B、pal4_g16、KV cache 512
- 投機的デコード: MTP drafter、verify width 4
- 試験質問: Swift の `square(_:)` 実装
- 出力: 24 トークン
- 数値は初回 Core ML コンパイルを含まない warm run

## 同一 iPhone 17 での A/B

| 構成 | 24 token | tok/s | 基準比 | ピークメモリ | 出力 |
|---|---:|---:|---:|---:|---|
| 従来: full 8層 CPU+GPU、drafter `.all` | 52.61秒 | 0.456 | 基準 | 2,369.3 MB | 基準 |
| full 8層を ANE | 42.38秒 | 0.566 | 19.5%短縮 / 24.2%高速 | 1,884.0 MB | 完全一致 |
| 最終: full 8層 ANE、drafter CPU+GPU、verify保持0 | 43.16秒 | 0.556 | 18.0%短縮 / 21.9%高速 | 665.6 MB | 完全一致 |
| verify 24層を常駐 | 47.55秒 | 0.505 | 最終構成より10.2%遅い | 842.8 MB | 完全一致 |

最速値だけなら drafter `.all` の 42.38 秒だが、起動直後に ANE コンパイル失敗とフォールバックが発生し、メモリも大きい。0.78 秒の差と引き換えにピークを約 1.2 GB削減できるため、drafter CPU+GPU を既定値にした。

## ログから分かった原因

最後の speculative sweep における verify 48層の内訳:

| 端末・構成 | モデルロード合計 | predict合計 | sweep合計 |
|---|---:|---:|---:|
| iPhone 14（従来） | 3.6576秒 | 0.5925秒 | 4.9861秒 |
| iPhone 17（従来） | 4.6822秒 | 0.5724秒 | 5.9371秒 |
| iPhone 17（full ANE） | 3.5851秒 | 0.3344秒 | 4.4521秒 |
| iPhone 17（最終） | 3.6958秒 | 0.3285秒 | 4.5143秒 |

iPhone 17 の実演算は従来構成でも iPhone 14 より約3%速かった。一方でロードが約28%遅く、その差が SoC の優位を隠していた。full-attention 層を ANE にすると、ロードと演算の両方が改善した。

verify モデルを24層常駐させる案も試したが、メモリ圧と実行計画の競合により逆に遅くなったため、常駐は診断用オプションだけ残して既定値を0にした。

## 実装した端末別ポリシー

- iPhone18,x 以降:
  - full-attention 層を ANE で実行
  - MTP drafter は CPU+GPU
  - verify モデル常駐は0
- iPhone 14を含む旧端末:
  - full-attention 8層は従来どおり CPU+GPU
  - 動作実績のある既存ポリシーを維持
- 環境変数で全項目を A/B 可能:
  - `COREML_PROBE_KV_FULL_ANE`
  - `COREML_PROBE_DRAFTER_COMPUTE`
  - `COREML_PROBE_RETAIN_VERIFY`

したがって、実装全体が iPhone 14 専用だったわけではない。iPhone 14向けの安全策1点がグローバル設定になっていたことが速度逆転の中心だった。

## 品質と安定性

- 4構成すべてで回答文字列が一致
- 全24 token IDが完全一致
- 画像入力後の生成も同じ KV decode 経路を使うため、出力側の最適化で精度は変化しない
- iPhone 17で full ANE の24トークン生成を複数回完走
- adaptive tree はこの質問では acceptance 0.944、top-1 hit 6/6 だったため発動せず、今回の改善幅には含まれない

## 初回コンパイルとストレージ上の注意

Core ML の初回実行は各モデルの実行計画を生成するため、warm run より大幅に遅い。今回の再署名・再インストール後は、過去の多数の A/B キャッシュが残った状態で `No space left on device` が発生した。専用アプリデータを消して再生成すると2トークンの初回ウォームアップは342.3秒で完走したが、ストレージが逼迫するとキャッシュが永続化せず、毎回再コンパイルになる。

製品利用時の速度指標には初回値を混ぜず、Core ML キャッシュ用の空き容量を確保した上で2回目以降を測る必要がある。

再接続後に Finder で確認した実空き容量は 23.66 GB（255.13 GB中）だった。この状態では、前半レイヤーの実行計画を保存すると後半レイヤーが未保存になり、次のリクエストでは逆に前半側が追い出される。再起動後の連続試験でも ANE/BNNS コンパイル時の `No space left on device` を再現した。

verify 36/48層をメモリ常駐させる低ストレージ向け構成も試した。常駐モデル自体は約250 MBで収まったが、そのロードによって LM head と embedding のキャッシュが追い出され、同じ容量不足が発生した。したがって `retainVerify=36` は採用しない。

現在の 48 prefill + 48 verify 分割をコールド起動から安定運用するには、端末空き容量の追加確保、またはモデル分割数を減らす再変換が必要になる。後者は複数層を融合した Core ML モデルの生成と再検証になるため、次段階では RunPod の GPU 変換環境を使うのが現実的。

試験終了時には端末容量を戻すため `CoreMLProbe` と専用キャッシュをアンインストールした。ソース、12 GBの最小ビルド、全計測ログは Mac 側に保存している。

## 6層融合モデルの追試（2026-07-25）

5 sliding-attention + 1 full-attention の周期を1本へ融合し、prefill 48本 + verify 48本を8本 + 8本へ減らす変換を実装した。重み、pal4_g16量子化、KV形状、target logitsは変更していない。

- PyTorch参照検証: `maxAbsErr=0.02344`、参照最大値 `16.78`、相対誤差 `0.00140`
- 先頭6層のprefill/verify: 各約651 MiB（Macコンパイル後）
- 入出力: 6層分のK/Vを絶対レイヤー名で受け渡し、Swift側は融合モデルがある範囲だけ自動使用
- 従来の単層モデルを残した部分融合アプリで、先頭6層だけを安全にA/B可能

iPhone 17で `.all` を使った初回ロードは、融合prefillの実行計画生成中にSIGKILLとなった。端末の診断はJetsamではなく `diskwrites_resource` で、フットプリントは約92 MiBのまま、Core ML/Espresso/BNNSが7秒間に1,073.76 MBを書き込み、1,073.74 MBの上限へ到達していた。したがって今回の停止原因は推論メモリ不足ではなく、大きな融合ANE実行計画の初回生成である。

融合グループだけをCPU+GPUへ切り替える `COREML_PROBE_FUSED_COMPUTE` を追加した。ANE単独で成立しない場合も、残りの単層モデルをANEに維持した混合構成でロード回数削減の効果を測れる。

### 融合数の実機A/B

再起動・再接続後、融合グループだけをCPU+GPU、残りの単層モデルをANEにした。従来と同じSwift問題、24 token、MTP linear verifyで比較した。

| 構成 | 24 token | tok/s | first token | speculative sweep平均 | ピーク | 全token ID |
|---|---:|---:|---:|---:|---:|---|
| 従来48分割 | 43.16秒 | 0.556 | - | 約4.51秒 | 665.6 MB | 基準 |
| 先頭6層だけ融合 | 37.56秒 | 0.639 | 10.45秒 | 4.52秒 | 1,691.7 MB | 完全一致 |
| 先頭12層を融合 | 39.57秒 | 0.607 | 11.43秒 | 4.69秒 | 1,768.2 MB | 完全一致 |

先頭6層だけの融合は従来比で時間を13.0%短縮し、tok/sを14.9%改善した。ただしCPU+GPUの大きな融合グラフによりピークメモリは約1.03 GB増えた。層6...11も融合すると6層版より5.4%遅くなり、メモリも76.5 MB増えたため採用しない。

層6...11だけを見ると、単層6本のprefillは0.6698秒、融合は1.3022秒、6 sweep分のverifyは単層2.5984秒、融合2.9659秒だった。全面融合に単調な速度向上はなく、追加RunPod変換の根拠もない。採用候補は先頭6層の2モデルだけとする。

### 画像・音声の現行KV経路

従来の `image-smoke` / `audio-smoke` 自動試験は旧Seq64 decoderを固定で要求していた。自動試験を通常チャットと同じKV経路へ接続し、音声もKV資産がある場合はSeq320を使うよう修正した。

- 画像: 合成UIImage → 256 patch embedder → hidden位置46...301へoverlay → 融合6層 + 単層42層 → LM head
  - warm 1 token: 9.4866秒
  - 同じ画像のtoken `#238618`を再現
  - ピーク1,635.4 MB、解析エラー0
- 音声: 非ゼロの決定的fixture → 32 token audio embedder → hidden位置270...301へoverlay → 同じKV経路
  - warm 1 token: 10.3396秒
  - token `#207330`
  - ピーク1,627.4 MB、解析エラー0

`scripts/prepare_ios_fused_kv_assets.sh` は、実測採用した層0...5のprefill/verifyだけをコンパイルまたはコピーし、iOSの `Models` に配置する。遅かった第2グループ以降は意図的に配置しない。

## 計測ファイル

`artifacts/coreml-performance/20260724-iphone17-rootcause/`

- `baseline-tree.json` / `.log`
- `full-ane-tree-warm.json` / `.log`
- `optimized-retain24-warm.json`
- `optimized-retain0-warm.json` / `.log`
- `final-clean-warmup2.json`
- `final-default-first-failed.log`
- `final-warm1-cache-starved.log`
- `post-reboot-cache-check2-complete.json` / `.log`
- `post-reboot-warm2-cache-pressure.log`
- `retain36-cache-pressure-first.log`

6層融合の部分実機試験:

`artifacts/coreml-performance/20260725-iphone17-fused6/partial-group00/`

- `smoke-2tok/device-console.log`
- `smoke-2tok-rerun/device-console.log`
- `diagnostics/CoreMLProbe.diskwrites_resource-2026-07-25-111755.ips`
- `mixed-24tok-warm/device-console.log`
- `image-kv-smoke-warm/device-console.log`
- `audio-kv-smoke/device-console.log`

12層融合の比較:

`artifacts/coreml-performance/20260725-iphone17-fused6/partial-group01/`

- `mixed-24tok-warm/device-console.log`
- `mixed-24tok-warm/analyze.log`
