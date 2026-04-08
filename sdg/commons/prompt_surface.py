from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class PromptBlock:
    key: str
    lines: tuple[str, ...]
    heading: str | None = None
    numbered: bool = False

    def render(self) -> str:
        if not self.lines:
            return ""

        lines = list(self.lines)
        if self.numbered:
            lines = [f"{index}. {line}" for index, line in enumerate(lines, start=1)]
        if self.heading:
            lines = [f"{self.heading}:", *lines]
        return "\n".join(lines)


@dataclass(frozen=True)
class SurfacePlan:
    key: str
    block_order: tuple[str, ...]


def render_prompt(blocks: dict[str, PromptBlock], plan: SurfacePlan) -> str:
    rendered: list[str] = []
    for key in plan.block_order:
        assert key in blocks, f"missing prompt block: {key}"
        text = blocks[key].render().strip()
        if text:
            rendered.append(text)
    return "\n\n".join(rendered)


def default_surface_axes() -> dict[str, tuple[str, ...]]:
    return {
        "surface_intro": ("context", "assignment", "task_first"),
        "surface_instruction": ("solve", "unique", "consistency"),
        "surface_answer": ("respond", "write", "complete"),
        "surface_clue": ("plain", "compact", "deductive"),
    }


def sample_surface_plan(rng: Random, axes: dict[str, tuple[str, ...]]) -> dict[str, str]:
    return {
        key: rng.choice(values)
        for key, values in axes.items()
    }
