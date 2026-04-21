from __future__ import annotations

from typing import Any


class ArticleMode:
    """Generate a realistic prompt that would lead an LLM to write the source article."""

    name = "backtranslation"

    def framing_messages(self, article: dict[str, Any]) -> list[dict[str, str]]:
        user_lines = []
        if article["title"]:
            user_lines.append(f"Article title: {article['title']}")
        user_lines.append("Article text:")
        user_lines.append(article["text"])

        return [
            {
                "role": "system",
                "content": (
                    "Write a short prompt that would lead an LLM to generate something like the above, "
                    "don't be too specific. "
                    "Write the prompt in the same language as the article. "
                    "If using the title helps avoid getting too specific, mention the title directly. "
                    "Example of a good response: "
                    "\"Skriv en kort, informativ artikel om \\\"<titel>\\\". "
                    "Beskriv kort baggrund, indhold eller betydning. "
                    "Hold stilen neutral, let encyklopædisk og overskuelig.\""
                ),
            },
            {
                "role": "user",
                "content": "\n".join(user_lines),
            },
        ]

    def assemble_prompt(self, framing: str, article: dict[str, Any]) -> str:
        return framing  # the LLM output is already the final prompt

    def target(self, article: dict[str, Any]) -> str:
        return article["text"]

    def meta_extra(self, article: dict[str, Any]) -> dict[str, Any]:
        return {}
