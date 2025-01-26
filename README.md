# ai-agent-without-framework

## 事前準備

パッケージ管理には [uv](https://github.com/astral-sh/uv) を使用しています。
uv をインストール後、`uv sync` で必要なライブラリをインストールしてください。

またこのレポジトリでは [OpenAI API](https://openai.com/index/openai-api/) を活用します。
各自で OpenAI API に登録して API キーを発行したのち、以下のようにして API キーを配置してください。

1. `sample.env` を `.env` という名前でコピーする
1. `.env` に含まれる `OPENAI_API_KEY="your_api_key_here"` を実際の API キーに置き換えてください。

## 各スクリプトの説明

### simple_chatbot.py

過去の会話履歴を保持したまま会話を行うチャットボットです。

```bash
uv run simple_chatbot.py
```

### simple_agent.py

検索エンジンを任意の回数活用して質問に答えるエージェントです。

```bash
uv run simple_agent.py
```
