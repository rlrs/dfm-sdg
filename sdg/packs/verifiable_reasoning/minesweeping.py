STARTER_GUIDE = {
    "family": "minesweeping_logic",
    "status": "scaffold",
    "dolci_subset": "minesweeping",
    "goal": "Generate exact Minesweeper deduction puzzles with visible clue numbers and a hidden mine mask.",
    "answer_contract": "mask_grid",
    "starting_scope": {
        "sizes": ["5x5", "6x6"],
        "clue_regimes": ["full clue board", "partially revealed clues"],
    },
    "next_steps": [
        "Represent boards with safe clue cells and hidden mine cells.",
        "Add an exact solver for mine placement from the visible number clues.",
        "Render the clue board directly in the prompt with a compact legend.",
        "Verify uniqueness and keep the final answer as a mine mask grid.",
    ],
}


def starter_guide() -> dict[str, object]:
    return dict(STARTER_GUIDE)
