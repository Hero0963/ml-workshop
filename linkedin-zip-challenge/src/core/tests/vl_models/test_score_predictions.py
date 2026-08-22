# src/core/tests/vl_models/test_score_predictions.py
"""P4c's held-out numbers come out of this module, so its arithmetic has to be right.

Everything here is offline: no Colab, no GPU, no Ollama. The one live dependency is
CP-SAT, exercised in a single test because the solvability column is otherwise
unverified.
"""

import json

import pytest

from src.core.vl_models.score_predictions import (
    LabelError,
    format_summary,
    path_is_legal,
    read_predictions,
    score_predictions,
    score_record,
    summarize,
    truth_from_label,
)

# A 3x3 board with a single wall: small enough to reason about by hand, large enough
# to have a waypoint, an empty cell and a wall all at once.
LAYOUT = [["01", "  ", "  "], ["  ", "  ", "  "], ["  ", "  ", "02"]]
WALL = {"cell1": [0, 0], "cell2": [0, 1]}
OTHER_WALL = {"cell1": [1, 1], "cell2": [1, 2]}


def make_label(
    layout: list[list[str]] | None = None, walls: list[dict] | None = None
) -> str:
    return json.dumps(
        {
            "layout": LAYOUT if layout is None else layout,
            "walls": [WALL] if walls is None else walls,
        }
    )


def make_record(raw_output: str, label: str | None = None, **extra) -> dict:
    return {
        "file_name": "images/000000.jpg",
        "label": make_label() if label is None else label,
        "raw_output": raw_output,
        **extra,
    }


class TestScoreRecord:
    def test_perfect_prediction_is_an_exact_match(self):
        result = score_record(make_record(make_label()), solve=False)

        assert result["json_parsed"] is True
        assert result["exact_match"] is True
        assert result["layout"]["cell_accuracy"] == 1.0
        assert result["layout"]["waypoint_recall"] == 1.0
        assert result["walls"]["f1"] == 1.0
        assert result["truth_wall_count"] == 1

    def test_markdown_fenced_output_is_accepted(self):
        """The model is asked for bare JSON but often fences it anyway."""
        result = score_record(make_record(f"```json\n{make_label()}\n```"), solve=False)
        assert result["exact_match"] is True

    def test_a_missed_wall_costs_recall_and_the_exact_match(self):
        result = score_record(make_record(make_label(walls=[])), solve=False)

        assert result["json_parsed"] is True
        assert result["exact_match"] is False
        assert result["walls"]["recall"] == 0.0
        assert result["layout"]["cell_accuracy"] == 1.0

    def test_a_hallucinated_wall_costs_precision_and_the_exact_match(self):
        """A false wall is as fatal as a missed one -- puzzle_03 failed exactly this way."""
        result = score_record(
            make_record(make_label(walls=[WALL, OTHER_WALL])), solve=False
        )

        assert result["exact_match"] is False
        assert result["walls"]["recall"] == 1.0
        assert result["walls"]["precision"] == 0.5

    def test_a_wall_between_non_neighbours_is_dropped_and_reported(self):
        far_apart = {"cell1": [0, 0], "cell2": [2, 2]}
        result = score_record(
            make_record(make_label(walls=[WALL, far_apart])), solve=False
        )

        assert result["exact_match"] is True
        assert result["walls"]["predicted_count"] == 1
        assert result["parser_warnings"]

    def test_non_json_output_counts_as_a_parse_failure(self):
        result = score_record(
            make_record("Let me think about this puzzle."), solve=False
        )

        assert result["json_parsed"] is False
        assert result["exact_match"] is False
        assert "parse_error" in result

    def test_a_wrong_grid_size_scores_zero_rather_than_partial_credit(self):
        small = [["01", "  "], ["  ", "02"]]
        result = score_record(make_record(make_label(layout=small)), solve=False)

        assert result["layout"]["shape_match"] is False
        assert result["layout"]["cell_accuracy"] == 0.0
        assert result["exact_match"] is False

    def test_a_wall_free_board_predicted_wall_free_scores_one(self):
        label = make_label(walls=[])
        result = score_record(make_record(label, label=label), solve=False)

        assert result["truth_wall_count"] == 0
        assert result["walls"]["f1"] == 1.0
        assert result["exact_match"] is True

    def test_carried_fields_survive_into_the_result(self):
        result = score_record(
            make_record(make_label(), generation_seconds=8.25), solve=False
        )
        assert result["file_name"] == "images/000000.jpg"
        assert result["generation_seconds"] == 8.25

    def test_a_broken_label_is_a_dataset_bug_not_a_model_miss(self):
        with pytest.raises(LabelError):
            score_record(make_record(make_label(), label="not json"), solve=False)

    def test_solvability_is_reported_when_asked(self):
        result = score_record(make_record(make_label()), solve=True)
        assert result["solvable"] is True


class TestSummarize:
    def test_wall_free_boards_do_not_flatter_the_walled_average(self):
        """Predicting zero walls on a wall-free board earns a free 1.0."""
        wall_free = make_label(walls=[])
        results = [
            score_record(make_record(wall_free, label=wall_free), solve=False),
            score_record(make_record(make_label(walls=[])), solve=False),
        ]
        summary = summarize(results)

        assert summary["mean_wall_f1"] == 0.5
        assert summary["mean_wall_f1_walled_only"] == 0.0
        assert summary["walled_items"] == 1

    def test_micro_rates_pool_every_wall(self):
        both = make_label(walls=[WALL, OTHER_WALL])
        results = [
            score_record(
                make_record(make_label(walls=[WALL]), label=both), solve=False
            ),
            score_record(make_record(make_label(walls=[WALL])), solve=False),
        ]
        summary = summarize(results)

        # 2 hits out of 2 predicted, 2 hits out of 3 true walls.
        assert summary["micro_wall_precision"] == 1.0
        assert summary["micro_wall_recall"] == pytest.approx(2 / 3)

    def test_exact_match_rate_counts_parse_failures_as_misses(self):
        results = [
            score_record(make_record(make_label()), solve=False),
            score_record(make_record("no json here"), solve=False),
        ]
        summary = summarize(results)

        assert summary["items"] == 2
        assert summary["json_parse_rate"] == 0.5
        assert summary["exact_match"] == 1
        assert summary["exact_match_rate"] == 0.5

    def test_breakdown_buckets_by_the_number_of_walls_on_the_board(self):
        wall_free = make_label(walls=[])
        results = [
            score_record(make_record(wall_free, label=wall_free), solve=False),
            score_record(make_record(make_label()), solve=False),
        ]
        summary = summarize(results)

        assert set(summary["by_wall_count"]) == {"0", "1"}
        assert summary["by_wall_count"]["1"]["items"] == 1
        assert summary["by_wall_count"]["1"]["exact_match_rate"] == 1.0

    def test_an_empty_set_summarises_to_zeroes_rather_than_dividing_by_zero(self):
        summary = summarize([])
        assert summary["items"] == 0
        assert summary["exact_match_rate"] == 0.0
        assert summary["mean_wall_f1_walled_only"] == 0.0

    def test_format_summary_renders_every_bucket(self):
        results = [score_record(make_record(make_label()), solve=False)]
        text = format_summary(summarize(results))

        assert "EXACT MATCH" in text
        assert text.strip().splitlines()[-1].split()[0] == "1"


class TestReadPredictions:
    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            json.dumps(make_record(make_label())) + "\n\n", encoding="utf-8"
        )
        assert len(read_predictions(path)) == 1

    def test_end_to_end_from_a_file(self, tmp_path):
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps(make_record(make_label())),
                    json.dumps(make_record(make_label(walls=[]))),
                ]
            ),
            encoding="utf-8",
        )
        results, summary = score_predictions(read_predictions(path), solve=False)

        assert len(results) == 2
        assert summary["exact_match"] == 1


# A 3x3 board with 01 at (0,0) and 02 at (2,2); the only wall sits between (0,0) and
# (0,1). Small enough to write legal and illegal routes out by hand.
LEGAL_PATH = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2)]
PATH_THROUGH_THE_WALL = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 1),
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 2),
]


def truth_puzzle(walls: list[dict] | None = None):
    return truth_from_label(make_label(walls=walls))


class TestPathIsLegal:
    def test_a_hand_checked_route_is_accepted(self):
        assert path_is_legal(truth_puzzle(), LEGAL_PATH) is True

    def test_a_route_crossing_a_wall_is_rejected(self):
        assert path_is_legal(truth_puzzle(), PATH_THROUGH_THE_WALL) is False

    def test_the_same_route_is_fine_when_that_wall_is_not_there(self):
        """The asymmetry the pipeline turns on.

        Predicting extra walls can only over-constrain the solver, so its answer stays
        legal on the real board. Missing a wall lets the solver walk straight through
        one -- and nothing downstream complains.
        """
        assert path_is_legal(truth_puzzle(walls=[]), PATH_THROUGH_THE_WALL) is True

    def test_a_route_that_skips_a_cell_is_rejected(self):
        assert path_is_legal(truth_puzzle(), LEGAL_PATH[:-1]) is False

    def test_a_route_that_visits_a_cell_twice_is_rejected(self):
        repeated = LEGAL_PATH[:-1] + [LEGAL_PATH[0]]
        assert path_is_legal(truth_puzzle(), repeated) is False

    def test_a_route_that_teleports_is_rejected(self):
        jumped = [(0, 0), (2, 2)] + LEGAL_PATH[2:]
        assert path_is_legal(truth_puzzle(), jumped) is False

    def test_a_route_not_starting_on_waypoint_one_is_rejected(self):
        assert path_is_legal(truth_puzzle(), LEGAL_PATH[::-1]) is False

    def test_waypoints_must_be_collected_in_order(self):
        ordered = [["01", "  ", "03"], ["  ", "  ", "  "], ["  ", "  ", "02"]]
        puzzle = truth_from_label(make_label(layout=ordered, walls=[]))
        # Reaches 03 at index 2, long before 02.
        early = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1)]
        assert path_is_legal(puzzle, early) is False

    def test_an_empty_route_is_rejected(self):
        assert path_is_legal(truth_puzzle(), []) is False


class TestSolutionValidity:
    def test_a_perfect_read_produces_a_valid_solution(self):
        result = score_record(make_record(make_label()), solve=True)
        assert result["solution_valid_on_truth"] is True
        assert result["solvable"] is True

    def test_an_unparseable_answer_is_not_a_valid_solution(self):
        result = score_record(make_record("no json"), solve=True)
        assert result["solution_valid_on_truth"] is False

    def test_summary_separates_silent_failures_from_loud_ones(self):
        results = [
            score_record(make_record(make_label()), solve=True),
            score_record(make_record("no json"), solve=True),
        ]
        summary = summarize(results)

        assert summary["solution_valid_on_truth"] == 1
        assert summary["solution_valid_rate"] == 0.5
        assert summary["solvable_but_wrong"] == 0
        assert summary["by_wall_count"]["1"]["solution_valid_rate"] == 0.5
