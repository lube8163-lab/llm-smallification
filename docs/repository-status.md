# リポジトリの整理 — 2026-10-18

## 管理範囲

このpublicリポジトリはGemma 4 12BのCore ML分割・量子化・iPhone実験の正本です。
既存のモデル配布先とリポジトリ名は維持します。

Qwen 3.6 / 3.8 27BはCore AIのstreamingランタイム、変換adapter、資産構成を共有するため、
既存のprivate `coreai-streaming-llm`でまとめて管理します。Gemmaのハーネスへ混在させたり、
privateの作業履歴をこのpublicリポジトリへ取り込んだりしません。

## 統合した作業

`codex/gemma4-iphone-multimodal-smoke`の
`70a62a1d5130223c5feb9167c6f2661a216093a8`までの13コミットをmainへ統合します。
画像・音声チャット、KV生成、MTP投機デコード、iPhone世代別の実行先選択、
速度・品質診断、モデル取得手順、認証付きローカルAPIの実装と記事が対象です。
それまでのmainの履歴と研究ブランチは保持します。

READMEの速度・品質の値は各日付の実験記録です。今回の整理では新しい実機測定や
モデル再変換を行っていません。iCloud上で未取得のローカルファイルについては、
ファイルサイズが既存Gitソースと一致していても内容の一致を断定しません。

## 今後の更新

`origin/main`から作業ブランチを作り、ソースをコミットし、チェック済みPRを統合します。
モデル、端末ログ、ビルド生成物、認証情報はGit管理外です。再現に必要な計測結果は
個別に内容を確認してドキュメントへ残してください。実機試験はREADMEの手順を参照してください。

モデルの公開先は引き続き
[lube8163/gemma-4-12b-coreml-iphone-practical-chat](https://huggingface.co/lube8163/gemma-4-12b-coreml-iphone-practical-chat)
です。この整理ではモデルの公開範囲を変更していません。
