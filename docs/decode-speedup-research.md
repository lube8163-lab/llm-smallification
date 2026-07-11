# KV decode 高速化 調査メモ（2026-07-11）

対象: iPhone 14 (A15, 6GB) / Gemma 4 12B pal4_g16 / KVキャッシュ経路
現状: warm decode **~4.3s/token**（retain0）

## 1. 実測プロファイル（決定的事実）

実機ログから warm decode 1トークンの内訳を分解した結果:

| 項目 | 時間 | 割合 |
|---|---:|---:|
| **decodeモデル load ×48** | **3.5–3.8s** | **82%** |
| 48層 predict（ANE/GPU 演算） | 0.6s | 14% |
| embedding (seq1) | 0.002s | ~0% |
| lm_head | 0.02s | ~0% |

**ボトルネックは演算ではなく、毎トークン 48 モデル（計 5.3GB）の load/release。**
load は ~0.66ms/MB とバイト数比例（mmap + ANE 重み転送）で、モデル分割の粒度を変えても
総バイトが同じなら合計 load 時間はほぼ変わらない（seq64 時代の 4-chunk 実測とも整合）。

## 2. 潰した選択肢

- **常駐 (retain)**: retain12 = 3.84s/tok（+11%）だがメモリ+40%。retain24 = 6.07s/tok に**悪化**
  （A15 ANE の常駐プラン contention、iPhone17/A19 では起きない）。retain48 = OOM。
  → A15 では常駐は解にならない（実測済み）。
- **チャンク併合**: load がバイト律速のため、48単層→12×4層にしても合計はほぼ不変。
  プラン設定の固定費削減分（小）のみ。full-attn 層はチャンク化すると ANE 拒否の前科あり。
- **1つの巨大 decode モデル（48層+MLState）**: 1回 load で済むが、
  ~17k op のグラフを A15 ANECCompile が通す見込みが薄い（full層は単層 seq512-key でも
  intermittent 失敗）。ハイリスク。

## 3. 本命: Gemma 4 公式 MTP による投機デコード

**この計測プロファイルは投機デコードと最高に相性が良い**:
検証（verify）は Kトークンをまとめて 1 回の 48 層スイープで処理できるため、
「1スイープ = ~4s」のコストを K トークン分に償却できる。

- Gemma 4 は**全サイズ MTP (Multi-Token Prediction) ヘッド付きで学習**されており、
  公式ドラフタが `google/gemma-4-12B-it-assistant` として配布されている
  （HF transformers の assisted generation / vLLM / llama.cpp 対応済み）。
- 別家系ドラフト（Gemma 3 270M/1B）は語彙数は同じ 262k だが tokenizer drift があり
  非推奨。公式 MTP ドラフタなら ID 完全一致。
- 我々の greedy (argmax) パイプラインなら受理判定は「target argmax == draft token」の
  最長プレフィックスで簡潔。

### 期待値

acceptance 2.5–3.5 tok/スイープ（公式ブログの ~3x 主張と整合）として:
**4.3s/token → 1.3–1.7s/token（約 3x）**

### 必要な作業

1. **ドラフタ入手・検証**（Mac で可能な見込み）
   - `google/gemma-4-12B-it-assistant` を DL、アーキテクチャ確認
     （MTPヘッド型: target の hidden を入力に取るか、スタンドアロン小型LMか）
   - 小型なら Mac の CPU torch + coremltools で Core ML 変換可（RunPod 不要）
2. **verify 用 decode グラフ変換**（RunPod 必要、~30分）
   - 既存 `runpod_convert_gemma4_coreml_kv.py` の decode を seq=K (4 or 8) で再変換
     （x[1,K,3840], mask[1,1,K,512+K], k/v_new[1,H,K,D] — スクリプトはほぼパラメータ化済み）
3. **Swift**: ドラフト→verify→受理/巻き戻しループ、KVスロットの一括書込/巻き戻し
4. 計測: acceptance 実測、体感速度

### リスク

- ドラフタが「target hidden 入力型」だと配線が増える（それでも可）
- acceptance がモバイル量子化モデルで想定より低い可能性（>2 なら実用益あり）
- seq=K decode グラフの ANE 適性は seq=1 と seq=320 の中間なので通る見込み大

## 4. 併用可能な軽い補完策

| 施策 | 期待効果 | コスト |
|---|---|---|
| pal3（3bit）decode 重み | load −25% (~0.9s/tok 短縮) | RunPod 再変換 + 品質確認 |
| lm_head 呼び出しの verify 統合 | 微小 | Swift のみ |

## 5. 推奨順序

1. ドラフタ checkpoint の中身確認（Mac、コスト小、全体の成否を左右）
2. OK なら RunPod 起動して seq=K verify グラフ変換 + ドラフタ変換
3. Swift 投機ループ実装 → 実機 acceptance 計測
4. 必要なら pal3 併用を追加検討
