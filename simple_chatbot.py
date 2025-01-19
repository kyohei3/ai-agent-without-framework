import openai
from dotenv import load_dotenv
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)

load_dotenv()


class SimpleChatbot:
    """会話履歴を保持してマルチターンの対話を行うチャットボット"""

    def __init__(self, system_prompt: str) -> None:
        self._client = openai.OpenAI()
        self._system_message: ChatCompletionMessageParam = {
            "role": "system",
            "content": system_prompt,
        }
        self._message_history: list[ChatCompletionMessageParam] = []

    def _get_response(self, user_query: str) -> str | None:
        """ユーザーからメッセージを受け取り、LLM により回答を生成する関数"""

        # ユーザの入力を message_history に追加
        user_message: ChatCompletionMessageParam = {
            "role": "user",
            "content": user_query,
        }
        self._message_history.append(user_message)

        # LLM を呼び出して回答を得る
        completion = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[self._system_message, *self._message_history],
        )

        # LLM からの回答を message_history に追加
        assistant_response = completion.choices[0].message
        if assistant_response.content:
            self._message_history.append(
                {"role": "assistant", "content": assistant_response.content}
            )

        return completion.choices[0].message.content

    def run(self) -> None:
        """チャットボットとの対話を開始"""
        while True:
            try:
                # ユーザーからの入力を待ち、LLM に問い合わせて回答を得る
                user_query = input("ユーザ: ")
                response = self._get_response(user_query)
                print(f"アシスタント: {response}")
            except KeyboardInterrupt:
                # Ctrl-C で終了
                break


def main() -> None:
    chatbot = SimpleChatbot("あなたは優秀なアシスタントです。")
    chatbot.run()


if __name__ == "__main__":
    main()
