---
title: "Gemma 4 12BをiPhoneで投機デコードする：2.4倍高速化とA19最適化"
emoji: "🚀"
type: "tech"
topics: ["ios", "coreml", "llm", "gemma", "swift"]
published: false
---

:::message
本記事はAI（Codex）の補助を受けて執筆しています。実装、モデル変換、実機計測は
実際に行い、本文の整理とレビューにAIを利用しました。
:::

## はじめに

前回の
[Gemma 4 12BをiPhone実機でマルチモーダルチャットにする](https://zenn.dev/lube8163/articles/55ec20f1c53315)
では、pal4量子化、ANE、KV cacheを組み合わせ、iPhone 14でテキスト・画像・音声を
完全オフライン処理できるところまで進めました。

残ったボトルネックは演算量ではありません。1 tokenごとに48個のdecoderモデルを
loadする時間でした。そこで今回は、前回「次に実装する」と書いたGemma 4公式MTP
（Multi-Token Prediction）drafterによる投機デコードを実装しました。

結論を先にまとめます。

- iPhone 14の高受理プロンプトでは、4.93秒/tokenから2.06秒/tokenへ**2.40倍**
- 別の低受理プロンプトでは4.20秒/tokenから4.12秒/tokenで、改善は**1.9%**
- targetのgreedy出力は、比較した全A/Bでtoken ID単位に一致
- iPhone 17の同じ24 token生成では、実行先の端末別最適化により総時間を
  52.61秒（0.456 token/s）から43.16秒（0.556 token/s）へ短縮
- さらに先頭6層だけを融合し、同じ24 tokenを総時間37.56秒、
  **0.639 token/s**まで短縮
- 画像・音声も通常チャットと同じKV・投機経路で回帰試験を通過

「投機デコードで常に2倍」ではありません。受理率、出力長、初回コンパイル、モデルの
load cacheによって大きく変わります。むしろ今回の収穫は、どこで効き、どこから先は
効かないかを実機で切り分けられたことでした。

## なぜ投機デコードが効くのか

従来のKV decodeは、1 tokenごとに48層を一巡します。

```text
token embedding
  → layer 0をload / predict / release
  → ...
  → layer 47をload / predict / release
  → lm_head
```

iPhone 14での内訳は、48層のpredictが約0.6秒なのに対し、モデルloadが
約3.5〜3.8秒でした。重みをすべてアプリのメモリへ常駐させるとメモリ圧や実行計画の
競合が悪化します。

投機デコードでは、小さなdrafterが複数tokenを予測し、targetは4 tokenをまとめた
verifyグラフで一度に検証します。今回のverify widthは4です。

```text
MTP drafter
  → draft 1, 2, 3

target verify [current, draft 1, draft 2, draft 3]
  → 一致したdraftを受理
  → 不一致位置はtarget tokenで訂正
```

targetが最終決定権を持つため、greedy decodeでは投機を無効にした場合と同じ出力に
できます。1回の48層loadで複数token進めば、今回の支配項だったload時間を受理token数で
償却できます。

## 実装したもの

### 1. 公式MTP drafter

`google/gemma-4-12B-it-assistant`の4層drafterを、固定shapeのCore MLグラフへ変換しました。
入力は現在tokenのembedding、target最終hidden、target側のsliding/full KV、
position、maskです。出力は262,144語彙のlogitsと、次のdraft stepへ渡すhiddenです。

ここで最も危険だったバグは、同じ3840次元である`token_emb`と`backbone_hidden`の
連結順でした。逆でも変換・実行は成功しますが、受理率が壊れます。Hugging Face実装と
同じ`[token_emb, backbone_hidden]`へ修正し、drafter本体をpal4、巨大なlm_headを
int8にしたmixed版を採用しました。

### 2. 4-token target verifier

従来のseq=1 decodeとは別に、48層それぞれへseq=4 verifyグラフを追加しました。
verifyは4行のhiddenと既存KVを受け取り、4行分のhiddenと新しいK/Vを返します。

Swift側では、受理前のK/Vを一旦未確定slotへ書き、targetが選んだ行だけを連続した
確定slotへ移します。却下されたtailはmaskされ、次roundで上書きされます。

### 3. adaptive top-3 tree

top-1 draftが外れ続ける場合、同じ4行verifyを1段の木として使います。

```text
row 0: current
row 1: candidate 1
row 2: candidate 2
row 3: candidate 3
```

各candidate行はrootと自分だけを参照します。targetがtop-3のどれかを選んでも、
target logitsから出力を確定するためgreedy品質は変わりません。ただし、候補が当たらない
プロンプトでは木を作るコストを回収できず、効果は小さくなります。

## 「2.06秒/token」と「4.20→4.12秒/token」の違い

同じ実装でも、質問が違います。

| iPhone 14試験 | 通常 | 投機 | 改善 | MTPの状態 |
|---|---:|---:|---:|---|
| 8-token高受理プローブ | 4.93秒/token | 2.06秒/token | **2.40倍** | acceptance 0.556、2.33 token/sweep |
| 24-token低受理プローブ | 4.20秒/token | 4.12秒/token | **1.9%短縮** | 外れが多くadaptive treeを使用 |

前者は8 tokenだけの選択的なprobeです。full runを含む別集計では5.19秒/tokenから
2.50秒/tokenで2.08倍でした。後者はより長く、MTPが予測しにくい質問です。

つまり2.06秒/tokenは「投機実装後の代表的な最良ケース」であり、4.12秒/tokenは
「低受理プロンプトを含む現実的な下限ケース」です。比較条件を外して数字だけ並べると
矛盾して見えますが、どちらも同じ実装の実測です。

## なぜ新しいiPhone 17が最初は遅かったのか

同じ品質試験をiPhone 14とiPhone 17で行うと、最初は新しいiPhone 17の方が遅いという
結果になりました。実装がiPhone 14専用だったわけではありませんが、iPhone 14での
Core ML実行計画エラーを避けるため、full-attention 8層をCPU+GPUへ逃がす設定が全端末へ
適用されていました。

iPhone 17（iPhone18,3、iOS 26.5.2）で同じSwift問題、24 tokenをA/Bした結果です。

| 構成 | 時間 | token/s | peak | 出力 |
|---|---:|---:|---:|---|
| full 8層 CPU+GPU、drafter `.all` | 52.61秒 | 0.456 | 2,369MB | 基準 |
| full 8層 ANE | 42.38秒 | 0.566 | 1,884MB | 完全一致 |
| 最終：full ANE、drafter CPU+GPU、verify常駐0 | 43.16秒 | 0.556 | 666MB | 完全一致 |

最速は42.38秒ですが、drafterのANEコンパイル失敗と大きなpeakがありました。そこで
約0.78秒と引き換えにdrafterをCPU+GPUへ固定し、peakを約1.2GB削減した43.16秒を
安定構成にしました。

現在は端末世代を見て既定値を分けています。

- iPhone18,x以降: full-attentionをANE、drafterをCPU+GPU
- iPhone 14を含む旧端末: full-attentionは実績のあるCPU+GPU
- verifyモデル常駐は両方とも0

各設定は環境変数で上書きできるため、将来のiOS/Core ML更新でも同一アプリから再評価できます。

## 48モデルのload回数を減らす

iPhone 17でも、最後まで残った支配項はverifyモデルのloadでした。そこでGemma 4の
`sliding × 5 + full × 1`という自然な6層周期を、1つのCore MLグラフへ融合しました。

まず先頭6層だけを融合し、残り42層は従来の単層モデルのままにしました。重み、
pal4量子化、KV shape、target logitsは変更していません。

| 構成 | 24 token | token/s | first token | peak | token ID |
|---|---:|---:|---:|---:|---|
| 従来48分割 | 43.16秒 | 0.556 | - | 666MB | 基準 |
| 先頭6層融合 | **37.56秒** | **0.639** | 10.45秒 | 1,692MB | 完全一致 |
| 先頭12層融合 | 39.57秒 | 0.607 | 11.43秒 | 1,768MB | 完全一致 |

先頭6層融合は時間を13.0%短縮し、token/sを14.9%改善しました。一方、第2グループまで
融合すると6層版より5.4%遅くなりました。融合数を増やせば単調に速くなるわけではなく、
大きなCPU+GPUグラフの実行時間とメモリがload削減を上回ります。

融合グラフをANEへ載せた初回試験はSIGKILLになりましたが、Jetsamではありませんでした。
診断ログは`diskwrites_resource`で、Core ML/Espresso/BNNSが7秒間に約1,074MBを書き、
OSの書き込み上限へ到達していました。そこで融合グループだけをCPU+GPU、残りの単層を
ANEにした混合構成を採用しています。

## 品質は落ちていないか

速度A/Bでは、通常decode、投機decode、端末別実行先、6層融合のすべてで、比較した
targetの出力文字列とtoken IDが一致しました。これは「元モデルの品質が高い」という意味では
なく、「今回の高速化がtarget greedy出力を変えなかった」という意味です。

別に小規模な品質診断も実施しました。

### iPhone 14

- 常識、論理、JSON、簡単なSwift、短答形式の抽象問題・翻訳・幻覚耐性は合格
- `17×23`を`419`と誤答し、target-onlyでも同じ結果
- 実PNGからRunPod、Pod状態、接続方法を認識
- 実WAVの“What is two plus two?”へ`4`と回答
- 自由回答は32 tokenでは前置き中に切れやすい

主評価は161 token、end-to-end 0.242 token/s、モデル生成区間0.287 token/sでした。
短い回答ではprefillやセットアップを償却できないため、投機probeのwarm decode値と
直接比較できません。

### iPhone 14とiPhone 17

同じ9問では回答本文とtoken IDが9/9で完全一致しました。端末変更による品質差は
観測できませんでした。出力上限を64〜96へ広げると翻訳、幻覚耐性、JSONは改善しましたが、
コードと抽象回答はまだ未完結になり、長いマルチモーダル生成ではCore ML呼び出しが
戻らない事象もありました。

これは20問程度の診断であり、MMLU等の統計的ベンチマークではありません。また、
算術誤答が上流Gemma 4、pal4量子化、Core ML変換のどこで生じたかは、同じpromptと
greedy設定をbf16/fp16参照モデルへ通さないと分離できません。

## 画像・音声も同じ経路で確認する

以前の`image-smoke` / `audio-smoke`は旧Seq64 decoderを固定で使っていました。これを
通常チャットと同じSeq320 prefill、KV、投機／融合経路へ接続しました。

- 画像: 256 patchをhidden位置46...301へoverlayし、warm 1 tokenを9.4866秒で生成
- 音声: 32 tokenをhidden位置270...301へoverlayし、warm 1 tokenを10.3396秒で生成
- どちらも解析エラーなし。既知fixtureのtokenを再現

ここで使った決定的fixtureは経路の回帰試験です。画像・音声の意味理解そのものは、前節の
実PNG・実WAV試験で別に確認しています。

## ストレージと初回コンパイル

モデル数が多い構成では、アプリ本体だけでなくCore MLが端末内に生成する実行計画も
ストレージを消費します。空き容量が少ないと、前半をcompileした後に後半が保存できず、
次回は別のcacheが追い出される状態になります。

今回の運用では次を徹底しました。

- cold compileとwarm runを分けて記録する
- テキスト、画像、音声を必要に応じて最小アプリへ分ける
- `.mlmodelc`を配布せず、`.mlpackage`をMac側でcompileする
- 実測で遅かった融合第2グループを同梱しない
- 端末の空き容量とCore ML cacheを計測条件へ含める

## 公開物

実装と再現スクリプト:

- [llm-smallification](https://github.com/lube8163-lab/llm-smallification)

変換済みCore MLパッケージ:

- [gemma-4-12b-coreml-iphone-practical-chat](https://huggingface.co/lube8163/gemma-4-12b-coreml-iphone-practical-chat)

Hugging Faceには従来のprefill/decode/endpoints/multimodalに加え、次を追加しています。

- 48個のseq=4 verify `.mlpackage`
- mixed pal4/int8 MTP drafter
- 実測採用した層00...05の融合prefill/verify

ダウンロードとSHA-256検証、Core ML compile、アプリへの配置はスクリプトで行えます。

```bash
HF_MODEL_PROFILE=speculative \
  ./scripts/download_hf_gemma4_coreml_models.sh
```

## まとめ

今回、iPhone上のGemma 4 12Bは「1 tokenずつ48モデルをloadする」構成から、MTPで
複数tokenを検証し、さらに実測で効果があった6層だけを融合する構成へ進みました。

高受理プロンプトの2.40倍は魅力的ですが、低受理では1.9%に留まります。現状の限界は
targetの演算性能より、drafterの受理率、Core MLモデルload、初回実行計画、ストレージ、
大きな融合グラフのCPU+GPUコストです。

それでも、target出力を変えず、画像・音声経路も維持したまま、iPhone 17で
0.639 token/sまで到達できました。「全部を融合する」「全部をANEへ載せる」ではなく、
端末とグラフごとに実測して混ぜることが、今回一番効いた最適化でした。
