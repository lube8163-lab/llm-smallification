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

## 6. 実装状況（2026-07-22）

アプリ側と変換側の投機デコード経路を実装した。実機で有効になる条件は、公式 MTP
ドラフタ1本と `seq=4` target verify bundle 48本が `Models/` に揃うこと。資産が欠ける
場合は従来の `seq=1` KV decode に自動フォールバックする。

1ラウンドの入出力は次の通り。

- ドラフタは target の layer 46（最後の sliding attention）と layer 47（最後の full
  attention）の KV、および「現在トークンの embedding + それを予測した target hidden」
  から3候補を自己回帰的に生成する。
- target verify は `[current, draft1, draft2, draft3]` の4入力を48層で一括処理する。
- greedy target と一致する draft の最長 prefix を受理する。不一致ならその位置の target
  token を補正として出し、3本すべて一致した場合は4行目の target token を bonus として出す。
- 未受理の verify KV は可視長へ加えないためマスクされ、次ラウンドで上書きされる。
- ログにはラウンドごとの draft/target ID、match 数、verify 秒数と、全体の
  `acceptance` / `tokensPerSweep` を残す。

RunPod で verify bundle を作るコマンド:

```bash
python scripts/runpod_convert_gemma4_coreml_kv.py \
  --target decode \
  --decode-seq 4 \
  --out-dir /workspace/gemma12b/coreml-layers-verify-seq4-pal4
```

ドラフタは全体 pal4 で語彙 head の argmax が変わったため、自動選択対象から外した。
既存の fp16 を正解基準として使えるほか、body を pal4、262k 語彙 head を int8 にする
mixed build を既定にした。

```bash
python scripts/runpod_convert_drafter_coreml.py \
  --quantization mixed \
  --keep-fp16 \
  --out-dir /workspace/gemma12b/coreml-drafter
```

Macへ転送後、コンパイル・検査・iOS appへの配置を一括実行する:

```bash
./scripts/prepare_ios_speculative_assets.sh \
  runpod-artifacts/speculative/verify \
  runpod-artifacts/speculative/drafter/gemma4_12b_drafter_step_kv512_mixed_pal4_g16_head_int8.mlpackage
```

実機検証では投機経路を必須にし、最低 tokens/sweep を指定できる。

```bash
./scripts/run_coreml_probe_device_automation.sh \
  --kind chat \
  --tokens 32 \
  --require-speculative \
  --min-speculative-tokens-per-sweep 2.0
```

`COREML_PROBE_DISABLE_SPECULATIVE=1` で従来経路との A/B が可能。ドラフタを明示する場合は
`COREML_PROBE_DRAFTER_MODEL` または `--drafter-model=` を使う。

### RunPod と iPhone 14 での検証結果（2026-07-23）

RunPod A100 で48本の `seq=4` verify bundle と mixed ドラフタを作成し、ローカルの
アセット検査で layer 0...47、KV形状、ドラフタ入出力の一致を確認した。ドラフタの
candidate-generator 入力は Transformers 公式実装と同じ
`[last_token_embedding, last_hidden_state]` の順で連結する必要がある。変換時に公式
forward と数値照合する検査も追加した。

iPhone 14（iPhone14,7 / iOS 26.5.2）で同じプロンプトから8トークンを生成し、通常経路と
投機経路の出力 ID がすべて一致することを確認した。

| 指標 | 通常 decode | 投機 decode |
|---|---:|---:|
| warm 秒/トークン | 4.93s | 2.06s |
| warm スループット | 0.203 tok/s | 0.486 tok/s |
| peak memory | 732.3 MB | 801.9 MB |

- 実測高速化率は **2.40x**。3ラウンドで draft 9本中5本が一致し、
  `acceptance=0.556`、`tokensPerSweep=2.33`。
- 比較では初回 Core ML コンパイルを含む外れ値（通常 token 2 と投機 round 1）を
  warm 平均から除外した。全 token の analyzer 平均でも 5.19s 対 2.50s の
  **2.08x** だった。
- 初回の 10.35s 対 7.48s（1.38x）は、実機自動化スクリプトが pal4
  norm+LM head まで CPU-only にする設定で測っていた。LM head を Core ML の `All`
  に戻すと 1回 3〜8s から約0.02sへ短縮した。本番UIは元から pal4 で `All`
  を既定にしており、自動化スクリプトもこれに揃えた。最初の CPU-only
  baseline 10.35s から見ると、現在の投機経路は **5.03x** 高速。
- A100 の Transformers assisted generation で5プロンプトの出力一致と合計
  **1.60x** の高速化も確認した。

24トークンの日本語説明プロンプトでは、固定3-draftの受理率が
`0.105`、`tokensPerSweep=1.21` まで落ち、warm は 4.40s/token、peak は
1,259.4 MB に達した。これへ対し、次の RunPod 不要の安全策を追加した。

- MTP が2ラウンド連続で全外れしたら、4ラウンドは target-only で冷却する。
- MTP の各ステップと LM head の行評価を明示的な `autoreleasepool` で囲み、
  受理しない中間 `MLMultiArray` をラウンド中に解放する。
- target-only では固定 seq=4 verify の行0のみを採用し、不要な LM head の行も
  計算しない。因果マスクにより greedy 出力は変わらない。

同じ24トークンの再試験で出力 ID は全て一致し、`targetOnly=10`、
warm 4.41s/token、peak 751.1 MB となった。速度は実質同等だが、peak を
**508.3 MB（40.4%）削減**できた。seq=4 verify 本体は target-only でも固定幅の
ため、このフォールバック単体に速度効果がないことも確認した。

full-attention 8層を明示的に CPU+GPU へ固定する既存モデルの試験は、
定常ラウンドが約5.21sから5.67sへ悪化したため不採用とし、デコーダは
Core ML の `All` 自動選択のままとした。

これにより、投機経路は資産作成、アプリ組み込み、実機での正しさ・速度・メモリ計測まで
完了した。RunPod Pod は必要ログと資産を回収後に Stop し、Terminate はしていない。
上記の LM head 実行先、適応制御、メモリ解放、CPU+GPU 比較はすべて
RunPod を再開せず、Mac と iPhone 14 のみで実施した。

### top-3 tree とマルチモーダル回帰（2026-07-23）

以降も RunPod は再開せず、Mac と iPhone 14 だけで次の3点を実装・検証した。

1. MTP の第1候補だけでなく top-3 と target の一致順位を各ラウンドに記録する。
2. 保存済み fp16 ドラフタを mixed ドラフタと実機 A/B する。
3. top-1 が2ラウンド連続で外れた場合、既存の `seq=4` verifier を
   `[current, candidate1, candidate2, candidate3]` の1段 tree として4ラウンド使う。

tree では root が cache + root、各候補が cache + root + 自分だけを見るマスクを使う。
target が選んだ候補の KV 行だけを連続スロットへ移すため、出力トークンは常に target の
greedy argmax で決まり、ドラフタの候補は速度にしか影響しない。

24トークンの同一日本語プロンプトを、Core ML のコンパイルキャッシュが温まった状態で
linear-chain と adaptive tree に通した結果:

| 指標 | linear + target-only | adaptive top-3 tree |
|---|---:|---:|
| warm 秒/トークン | 4.20s | 4.12s |
| verifier rounds | 20 | 19 |
| target-only / tree rounds | 10 / 0 | 0 / 9 |
| top-1 recall | 3/10 | 5/19 |
| top-3 recall | 5/10 | 7/19 |
| tree hit | - | 2/9 |
| tokens/sweep | 1.15 | 1.21 |
| peak memory | 737.5 MB | 842.7 MB |

両者の24出力 ID は完全一致した。adaptive tree はこの低受理プロンプトで
warm 秒/トークンを **1.9%短縮**し、スイープ数を **5%削減**、
tokens/sweep を **5.2%改善**した。一方でピークメモリは105.2 MB増えるため、
効果は小さいが精度を変えない補助策という位置づけになる。top-1 が良好な8トークンの
プロンプトでは tree は発火せず、従来どおり `tokensPerSweep=2.33`、出力 ID完全一致だった。

保存済み fp16 ドラフタは shape/型の検査には合格するが、実機では
`top1=0/8`、`top3=0/8`、`matched=0/24` となり、8トークン目から mixed の出力と
分岐した後に token 0 の連続も発生した。この資産は checkpoint/hidden semantics が
現 target と互換でないため不採用とし、自動選択から外した。fp16 という精度形式自体を
否定する結果ではなく、当該保存資産の互換性問題である。

画像・音声は embedder が仕様上 fp16 出力でも実機では float32 を返す場合があったため、
両経路とも overlay 前に fp16 へ明示変換するよう統一した。実機結果は次の通り。

| 経路 | 入力・範囲 | 結果 | 推論時間 | peak memory |
|---|---|---|---:|---:|
| image smoke | feature fixture → 48層 → LM head | token 236770、警告なし | 106.26s（cold load込み） | 256.1 MB |
| image chat | 実PNG、256 patches、seq320 KV | 応答先頭「この」(8978) | 161.65s（cold prefill load込み） | 415.4 MB |
| audio smoke | feature fixture → 48層 → LM head | token 236771、警告なし | 4.83s | 41.0 MB |
| audio chat | 実WAV 16kHz/mono/I16、32 frames | 応答先頭「iPhone」(50668) | 8.09s | 89.3 MB |

いずれも iPhone 14 上で embedder、hidden overlay、全48層、LM head、トークン生成まで
完走した。1トークン回帰なので回答品質の評価ではないが、画像・音声を含む本番相当の
入出力経路が壊れていないことを確認できた。
