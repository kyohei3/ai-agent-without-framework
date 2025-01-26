import enum
import json
from typing import Any

import duckduckgo_search
import openai
import pydantic
from dotenv import load_dotenv
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)

load_dotenv()

_SYSTEM_PROMPT = """あなたはユーザーの疑問に対して真摯に回答するアシスタントです。
ユーザーの質問に対してあなたのもつ事前知識で回答できる場合は、そのまま回答を生成してください。
直近のニュースや最近の出来事など、事前知識にない質問に対しては検索エンジンを活用して回答を生成してください。

回答の生成に検索結果を利用した場合は、必ず回答のその箇所に脚注を利用して参考にしたURL明示してください。
脚注は、本文中に [^1] のように数字を使って記述し、文末に以下のように実際のURLを記載してください。

[^1]: https:/...

また、現在の日付は2025-01-26です。
"""


@enum.unique
class State(enum.Enum):
    """エージェントの状態の一覧"""

    START = enum.auto()
    """ターンの開始"""

    LLM_CALL = enum.auto()
    """ LLM による回答生成またはツールの呼び出し"""

    TOOL_RUN = enum.auto()
    """ツールの実行"""

    END = enum.auto()
    """ターンの終了"""


class ToolCall(pydantic.BaseModel):
    """LLM によるツール呼び出しの情報を保持するクラス"""

    id: str
    type: str
    function_name: str
    arguments: dict[str, Any]


class SimpleAgent:
    """必要に応じて検索エンジンを活用して回答してくれるAIエージェント"""

    def __init__(self, system_prompt: str) -> None:
        self._client = openai.OpenAI()
        self._system_message: ChatCompletionMessageParam = {
            "role": "system",
            "content": system_prompt,
        }
        self._message_history: list[ChatCompletionMessageParam] = []

        # エージェントの内部状態を管理する変数
        self._state: State = State.START

    def _get_response(self, user_query: str | None) -> str | list[ToolCall] | None:
        """ユーザーからメッセージを受け取り、LLM により回答を生成する関数

        Args:
            user_query (str | None): ユーザーからの入力

        Returns:
            str | list[ToolCall] | None: LLM による回答。回答がテキストの場合は str、
                ツール呼び出しの場合は ToolCall のリスト形式の値を返す。
        """

        # ユーザの入力を message_history に追加
        if user_query:
            user_message: ChatCompletionMessageParam = {
                "role": "user",
                "content": user_query,
            }
            self._message_history.append(user_message)

        # LLM を呼び出して回答を得る
        completion = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[self._system_message, *self._message_history],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": (
                            "ウェブを検索し情報を取得するツール。"
                            "最近のニュースや出来事を参照する場合にはこのツールを使ってください。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "検索クエリ。検索クエリにはユーザーが使用している言語と同じ言語を用いてください。",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
        )

        # LLM からの回答を message_history に追加
        message_to_add: ChatCompletionAssistantMessageParam = {"role": "assistant"}
        assistant_response = completion.choices[0].message
        if assistant_response.content:
            message_to_add["content"] = assistant_response.content
        if assistant_response.tool_calls:
            message_to_add["tool_calls"] = [
                {
                    "id": t.id,
                    "type": t.type,
                    "function": {
                        "name": t.function.name,
                        "arguments": t.function.arguments,
                    },
                }
                for t in assistant_response.tool_calls
            ]
        self._message_history.append(message_to_add)

        # ツール呼び出しがあれば、呼び出し情報をToolCall のリストに変換して返す
        if assistant_response.tool_calls:
            result: list[ToolCall] = []
            for tool_call in assistant_response.tool_calls:
                result.append(
                    ToolCall(
                        id=tool_call.id,
                        type=tool_call.type,
                        function_name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                    )
                )
            return result

        # ツール呼び出しがなければ、レスポンスのテキストをそのまま返す
        return completion.choices[0].message.content or ""

    def _run_tool(self, tool_call: ToolCall) -> str:
        """LLM の出力に従ってツールを実行する関数

        Args:
            tool_call (ToolCall): LLM が指定した呼び出すツールや引数などの情報

        Returns:
            str: ツールの実行結果
        """
        if tool_call.function_name != "search":
            raise ValueError(f"Unsupported tool: {tool_call.function_name}")

        if "query" not in tool_call.arguments:
            raise ValueError("query argument is required for search tool.")

        # ツールの実行状況をユーザーに通知
        print(f"ツール呼び出し: {tool_call.function_name}({tool_call.arguments})")

        with duckduckgo_search.DDGS() as ddgs:
            results = ddgs.text(
                keywords=tool_call.arguments["query"],
                region="jp-jp",
            )

            # 検索結果を文字列に変換し、会話履歴に追加する
            tool_response = "\n\n".join(
                f"Title: {d['title']}\nURL: {d['href']}\nBody: {d['body']}"
                for d in results
            )
            self._message_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_response,
                }
            )
            return tool_response

    def run(self) -> None:
        """チャットボットとの対話を開始"""
        user_query: str | None = None
        while True:
            try:
                match self._state:
                    case State.START:
                        # START 状態の場合はユーザーからの入力を受け取り、LLM_CALL 状態に遷移
                        user_query = input("ユーザ: ")
                        self._state = State.LLM_CALL

                    case State.LLM_CALL:
                        # LLM_CALL 状態では LLM のレスポンスに応じて遷移先の状態が変わる
                        response = self._get_response(user_query)
                        user_query = None

                        if isinstance(response, str):
                            # テキストの場合は END へ遷移
                            self._state = State.END

                        elif isinstance(response, list):
                            # ツール呼び出しの場合は TOOL_RUN へ遷移
                            self._state = State.TOOL_RUN

                    case State.TOOL_RUN:
                        if not isinstance(response, list):
                            raise ValueError("response must be a list of ToolCall.")

                        # ツール呼び出しを実行・実行結果を履歴に保存したのち、再度 LLM_CALL 状態に遷移
                        for tool_call in response:
                            self._run_tool(tool_call)
                        self._state = State.LLM_CALL

                    case State.END:
                        if not isinstance(response, str):
                            raise ValueError("response must be a string.")

                        # LLM の回答表示して会話1往復が終了
                        # START 状態に戻り、ユーザーからの次の入力を待つ
                        print(f"アシスタント: {response}")
                        self._state = State.START

            except KeyboardInterrupt:
                # Ctrl-C で終了
                break


def main() -> None:
    agent = SimpleAgent(system_prompt=_SYSTEM_PROMPT)
    agent.run()


if __name__ == "__main__":
    main()
