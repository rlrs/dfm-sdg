from __future__ import annotations

from typing import Any

from sdg.packs.backtranslation.modes.article import ArticleMode
from sdg.packs.backtranslation.modes.eur_lex_sum import EurLexSumMode


def mode_for_source(source: dict[str, Any]) -> ArticleMode | EurLexSumMode:
    if source.get("summary_field"):
        return EurLexSumMode()
    return ArticleMode()


__all__ = ["ArticleMode", "EurLexSumMode", "mode_for_source"]
