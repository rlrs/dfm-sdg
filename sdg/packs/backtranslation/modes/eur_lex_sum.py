from __future__ import annotations

import re
from typing import Any

_STYLE_SEEDS = [
    # structure: request-first
    "Stil: direkte kommando, anmodning først. Eks: 'Opsummer følgende tekst:\\n\\n{{DOKUMENT}}'",
    "Stil: høflig anmodning, anmodning først. Eks: 'Vil du hjælpe mig med at opsummere dette? {{DOKUMENT}}'",
    "Stil: kontekstuel indledning, anmodning først. Eks: 'Jeg har et EU-dokument jeg gerne vil have et overblik over.\\n\\n{{DOKUMENT}}\\n\\nKan du give mig et resumé?'",
    "Stil: konkret resumé-anmodning med længdebegrænsning, anmodning først. Eks: 'Kan du give mig et kort resumé af nedenstående på tre til fem sætninger?\\n\\n{{DOKUMENT}}'",
    "Stil: faglig/formel, anmodning først. Eks: 'Udarbejd venligst et resumé af følgende juridiske dokument:\\n\\n{{DOKUMENT}}'",
    # structure: document-first
    "Stil: dokumentet først, kort afsluttende spørgsmål. Eks: '{{DOKUMENT}}\\n\\nHvad handler dette om?'",
    "Stil: dokumentet først, opsummering bedt om til sidst. Eks: '{{DOKUMENT}}\\n\\nKan du opsummere det ovenstående for mig?'",
    "Stil: dokumentet først, specifik resumé-anmodning til sidst. Eks: '{{DOKUMENT}}\\n\\nKan du skrive et kort resumé af ovenstående?'",
    "Stil: dokumentet først, uformel/nysgerrig tone. Eks: '{{DOKUMENT}}\\n\\nHvad er essensen her?'",
    # structure: document embedded mid-message
    "Stil: dokument midt i, omgivet af kontekst. Eks: 'Jeg sidder med dette EU-dokument:\\n\\n{{DOKUMENT}}\\n\\nKan du give mig et hurtigt overblik?'",
    "Stil: dokument midt i, med specifik opgaveformulering bagefter. Eks: 'Her er teksten:\\n\\n{{DOKUMENT}}\\n\\nSkriv et resumé på maksimalt 5 sætninger.'",
    # terse / minimal
    "Stil: meget kort og direkte, ingen kontekst. Eks: 'Resumér:\\n\\n{{DOKUMENT}}'",
]

_SECTION_HEADER_RE = re.compile(r"^(HVAD\b|INTRODUKTION\b|HOVEDPUNKTER\b)")


class EurLexSumMode:
    """Generate a varied summarization request; use the reference summary as target.

    Designed for EUR-Lex summary pairs where the source is a long legal document
    and the target is a structured summary. The boilerplate header (title, 'RESUMÉ AF:'
    marker, citation lines, and the first ALL-CAPS section heading) is stripped so the
    target starts directly with prose.
    """

    name = "summarization"

    def framing_messages(self, article: dict[str, Any]) -> list[dict[str, str]]:
        style_seed = _STYLE_SEEDS[article["row_index"] % len(_STYLE_SEEDS)]
        return [
            {
                "role": "system",
                "content": (
                    "Du skriver brugerbesked-skabeloner til en dansk opsummeringsassistent. "
                    "Brugeren har et langt EU-juridisk dokument på dansk og ønsker det opsummeret.\n\n"
                    "Skriv én realistisk, naturlig besked en bruger kunne sende. "
                    "Placer markøren {{DOKUMENT}} præcis det sted i beskeden, hvor dokumentteksten skal indsættes. "
                    "Følg den angivne stil, men vær fri til at variere den præcise ordlyd. "
                    "Skriv KUN beskedskabelonen med {{DOKUMENT}}-markøren, intet andet."
                ),
            },
            {
                "role": "user",
                "content": style_seed,
            },
        ]

    def assemble_prompt(self, framing: str, article: dict[str, Any]) -> str:
        document_text = article["text"]
        if "{{DOKUMENT}}" in framing:
            return framing.replace("{{DOKUMENT}}", document_text)
        # fallback: LLM used wrong placeholder or omitted it — append document
        return framing.rstrip() + "\n\n" + document_text

    def target(self, article: dict[str, Any]) -> str:
        return _strip_summary_header(article["summary"])

    def meta_extra(self, article: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": "summarization",
            "document_chars": len(article["text"]),
        }


def _strip_summary_header(text: str) -> str:
    """Strip the boilerplate header from EUR-Lex summary texts.

    EUR-Lex summaries begin with a metadata block: a title line, a
    'RESUMÉ' / 'RESUMÉ AF:' marker, and one or more legal citation lines.
    The substantive content starts at the first all-caps section header
    (e.g. 'HVAD ER FORMÅLET MED...', 'INTRODUKTION', 'HOVEDPUNKTER').
    That header line itself is also dropped so the target begins with prose.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _SECTION_HEADER_RE.match(line.strip()):
            return "\n".join(lines[i + 1:]).strip()
    return text.strip()
