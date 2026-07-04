---
title: "Gemma 4 12B UnifiedをCore ML化して、低メモリiPhoneでマルチモーダル経路だけ通す"
emoji: "📱"
type: "tech"
topics: ["ios", "coreml", "llm", "gemma", "swift"]
published: false
---

## これは何

Gemma 4 12B Unified 系の「text / image / audio を扱える」という性質を、できるだけ保ったまま iPhone 実機へ載せる実験をした。

結論からいうと、実用速度のチャットとしては厳しい。一方で、**実際の Gemma 4 image/audio embedder を Core ML 化し、その出力を 48 層 decoder + norm/lm_head まで iPhone 実機で通す** ところまでは確認できた。

つまり「フル品質・フル速度」ではなく、**低メモリ iPhone でもマルチモーダル対応モデルの経路が一応動いた** という到達点の記録である。

![Gemma 4 12B Core ML multimodal smoke path](./assets/gemma4-coreml-iphone-multimodal-path.svg)

## 実験環境

- Model: `google/gemma-4-12B-it-qat-q4_0-unquantized`
- iPhone: `iPhone18,3`, iOS `26.4.2`
- App: SwiftUI + Core ML
- Core ML shape: fixed `Seq64`
- Decoder: 48 layers, 4-layer chunks x 12
- KV cache: なし
- Quantization: int4 per-block, block size 32
- Runtime setting: endpoint `CPU`, decoder `All`
- Cache policy: run end
- Retain decoders: 0

変換は RunPod の A100 SXM 80GB で行い、ローカル Mac で `.mlpackage` を `.mlmodelc` に compile して iOS app に同梱した。

## まず分かったこと

12B を iPhone にそのまま載せる場合、速度の主なボトルネックは推論演算そのものよりも、分割した Core ML bundle の load / release だった。

安定構成では 8 token 生成が以下のような状態になった。

| 項目 | 結果 |
|---|---:|
| tokens | 8 |
| warm avg | 13.29 sec/token |
| peak memory | 587.7 MB |
| generated IDs | `#85141,#236924,#52119,#12553,#237221,#66447,#237669,#237230` |

decoder の予測自体は比較的短く、毎 token の大半は 12 個の decoder chunk を読む時間に使われている。

## 構成

既存の text path はこうなっている。

```text
input_ids
  -> text embedding Core ML
  -> hidden [1,64,3840]
  -> 48 decoder layers
  -> norm + lm_head
  -> logits
```

今回の multimodal smoke は、この `hidden [1,64,3840]` の seam に image/audio embedder の出力を差し込む形にした。

```text
pixel_values + image_position_ids
  -> Gemma4 embed_vision Core ML
  -> image_hidden [1,32,3840]
  -> hidden [1,64,3840] の後半へ挿入
  -> 48 decoder layers
  -> norm + lm_head
```

```text
input_features
  -> Gemma4 embed_audio Core ML
  -> audio_hidden [1,32,3840]
  -> hidden [1,64,3840] の後半へ挿入
  -> 48 decoder layers
  -> norm + lm_head
```

## Image smoke

変換した image embedder の Core ML contract は以下。

| input/output | shape |
|---|---|
| `pixel_values` | `fp32 [1,32,6912]` |
| `image_position_ids` | `int32 [1,32,2]` |
| `image_hidden` | `fp16 [1,32,3840]` |

実機結果:

| 項目 | 結果 |
|---|---:|
| image embedder load | 0.2879 sec |
| image embedder predict | 0.0161 sec |
| peak memory | 581.8 MB |
| generated token | `#258883` |
| token total | 34.4571 sec |
| analyzer | 0 errors / 0 warnings |

この時点で、実際の Gemma 4 image embedder Core ML endpoint の出力を decoder に渡し、norm/lm_head まで完走できた。

## Audio smoke

audio embedder はさらに小さい。

| input/output | shape |
|---|---|
| `input_features` | `fp32 [1,32,640]` |
| `audio_hidden` | `fp16 [1,32,3840]` |

実機結果:

| 項目 | 結果 |
|---|---:|
| audio embedder load | 0.0389 sec |
| audio embedder predict | 0.0027 sec |
| peak memory | 581.9 MB |
| generated token | `#236770` |
| token total | 33.1126 sec |
| analyzer | 0 errors / 0 warnings |

audio 側も同じ decoder / norm+lm_head 経路を通せた。

## なぜここを「限界点」と見ているか

一番大きい制約は、標準の image processor が現行の Seq64 に収まらないことだった。

RunPod 上で確認したところ、`<|image|>` を使う通常の processor 経路では、小さい PIL image を渡しても 256 image tokens になり、`pixel_values` は `[1,280,6912]` になった。今回の iPhone app は fixed Seq64 なので、そのままでは入らない。

また、48 層を減らすと速度は改善しうるが、Gemma 4 12B としての意味が薄くなる。decoder chunk を 8-layer に広げる実験もしたが、iPhone 側の execution-plan compilation で失敗するため不採用にした。

結果として、今回の実験はこう整理できる。

- 12B を低メモリ iPhone で実用チャットにするのは厳しい
- 速度の主因は decoder bundle の load/release
- 48 層 + 4-layer chunk + Seq64 が現時点の安定ライン
- image/audio の実 embedder を通す feasibility smoke は成立
- full image/audio prompting には Seq/token budget の再設計が必要

## やってよかったこと

今回の実験で一番効いたのは、いきなり full multimodal prompt を目指さず、decoder 入力 seam を固定して段階的に確認したことだった。

1. text-only 48-layer decoder を安定させる
2. norm+lm_head endpoint を正しくする
3. synthetic hidden を decoder に流す
4. real image embedder を流す
5. real audio embedder を流す
6. 最後に text regression で既存経路が壊れていないことを確認する

この順番にしたことで、どこで壊れたかを見失わずに済んだ。

## 最終的な感想

「Gemma 4 12B Unified を iPhone で普通に使う」という意味では、かなり厳しい。

ただ、「マルチモーダル対応モデルの一部を小さく切り出し、低メモリ端末上で実際の Core ML 経路として成立させる」という意味では、十分に面白い結果になった。

特に image/audio embedder は decoder 本体に比べるとかなり軽く、実機上でも load/predict は小さい。問題はその先の 48-layer decoder をどう扱うかであり、ここが 12B on-device の本丸だった。

次にやるなら、12B の速度改善をさらに追うより、以下のどちらかが現実的だと思う。

- real image/audio preprocessing を小さい smoke input へつなぐ
- もっと小さい multimodal model で実用 UX を狙う

今回の到達点は、後者へ進む前の「12B ではどこまで粘れるか」のよい境界線になった。
