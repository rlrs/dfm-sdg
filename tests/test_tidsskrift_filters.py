from __future__ import annotations

from sdg.packs.tidsskrift import build as tidsskrift_build


def test_score_paragraph_rejects_bibliographic_like_targets() -> None:
    bibliography_paragraphs = [
        "Nissen, M. A., 2006: Behandlerblikket - risiko og refleksion i socialt arbejde med børn og unge . In: Nordisk udkast, nr. 1, 2006, 61-80.",
        "Rasmussen, K., 2018: Visuel etnografi - om kendetegn, erfaringer og muligheder ved et metodisk felt i udvikling. In: Jacobsen, M. J. & Jensen, H. L. (red.): Etnografier (pp. 233-262) . København: Hans Reitzels Forlag.",
        "Jacobsen, M.H., Jørgensen, A. & Svendsen-Tune, S., 2010: Undersøgelser af udsatte og udstødte mennesker. In: Jacobsen, M.H., Kristiansen, S. & Prieur, A. (red.): Liv, fortælling, tekst - strejftog i kvalitativ sociologi . Aalborg: Aalborg Universitetsforlag.",
        "Ejbye-Ernst, N., Moss, B., Stokholm, D., Lassen, B., Praestholm, S. & Frøkjaer, T. (2019). Forskningsoversigt: Betydningen af dagtilbudsarbejde med børn i naturen - En forskningsoversigt med fokus på nordisk litteratur undersøgt med sigte på danske forhold. Center for børn og natur.",
        "Prins, K., Christensen, K. S., Thingstrup, S. H. & Aabro, C. (in press). Pædagogers lighedsskabende arbejde i daginstitutionen - dilemmaer og blinde vinkler. Barn .",
        "Rasch-Christensen, A. (2019) Om baggrunden for den styrkede pædagogiske læreplan. I: Aabro, C. (red.) Den styrkede pædagogiske læreplan. baggrund, perspektiver og dilemmaer (s. 27-41). Samfundslitteratur.",
        "Ringer, A., & Holen, M. (2021). Discursive Ethnography: Understanding psychiatric discourses and patient positions through fieldwork. I: Brookes, G. & D. Hunt (red.) Analysing Health Communication: Discourse Approaches. Palgrave Macmillan. s. 189-213.",
    ]

    for paragraph in bibliography_paragraphs:
        assert tidsskrift_build._score_paragraph(paragraph) == 0.0


def test_score_paragraph_keeps_normal_prose() -> None:
    prose = (
        "Analysen tager udgangspunkt i feltarbejde blandt unge i kommunale indsatser, "
        "hvor hverdagspraksis og relationer mellem professionelle og familier undersøges "
        "gennem længerevarende observationer. Artiklen viser, at tillid opbygges gradvist "
        "gennem konkrete handlinger, og at små justeringer i sagsbehandlerens tilgang "
        "kan ændre samarbejdet markant over tid."
    )

    assert tidsskrift_build._score_paragraph(prose) > 0.0


def test_target_is_not_references_rejects_single_entry_bibliography() -> None:
    row = {
        "target": (
            "Rasch-Christensen, A. (2019) Om baggrunden for den styrkede pædagogiske læreplan. "
            "I: Aabro, C. (red.) Den styrkede pædagogiske læreplan. baggrund, perspektiver og dilemmaer "
            "(s. 27-41). Samfundslitteratur."
        )
    }

    assert tidsskrift_build._target_is_not_references(row) is False
