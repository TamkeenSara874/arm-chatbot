"""Unit tests for the reply-status detection fast path.

Regression coverage for two real bugs found via live QA testing:
1. ReviewChunkMeta has no reply/response-status column at all (that's an
   AIO-dashboard-only concept), but "how many reviews haven't I replied to?"
   was silently answered with an unrelated exact number (total review count,
   or a sentiment-filtered count) at confidence 1.0 via the count_query fast
   path, instead of honestly saying this data isn't available.
2. The first fix attempt used rigid substring matching ("haven't replied")
   and missed natural question-word-order phrasing like "reviews haven't I
   replied to yet?", where a pronoun sits between the auxiliary and the verb
   -- confirmed live before switching to the current regex-based approach.
"""

from src.api.routes.chat import REPLY_STATUS_ANSWER, _is_reply_status_question


class TestIsReplyStatusQuestion:
    def test_havent_replied_simple(self) -> None:
        assert _is_reply_status_question("How many reviews haven't I replied to yet?")

    def test_havent_with_pronoun_between_auxiliary_and_verb(self) -> None:
        # The exact live-discovered bug: a pronoun between "haven't" and the
        # verb broke a naive "haven't replied" substring match.
        assert _is_reply_status_question("Reviews haven't I replied to are piling up.")

    def test_still_havent_responded(self) -> None:
        assert _is_reply_status_question(
            "Show me my negative reviews that I still haven't responded to."
        )

    def test_unanswered(self) -> None:
        assert _is_reply_status_question(
            "How many 1-star reviews are sitting there unanswered right now?"
        )

    def test_unanswered_in_a_different_sentence_shape(self) -> None:
        assert _is_reply_status_question(
            "Which unanswered reviews should I prioritize responding to first and why?"
        )

    def test_reply_status_phrase(self) -> None:
        assert _is_reply_status_question("What is my reply status on this?")

    def test_response_status_phrase(self) -> None:
        assert _is_reply_status_question("Can you show response status for recent reviews?")

    def test_case_insensitive(self) -> None:
        assert _is_reply_status_question("HAVEN'T I REPLIED TO THESE YET?")

    def test_not_yet_responded(self) -> None:
        assert _is_reply_status_question("Which reviews have not yet responded to?")

    def test_unreplied_word(self) -> None:
        assert _is_reply_status_question("Show me all unreplied reviews.")

    def test_negative_count_question_is_not_flagged(self) -> None:
        assert not _is_reply_status_question("How many negative reviews do I have?")

    def test_food_quality_question_is_not_flagged(self) -> None:
        assert not _is_reply_status_question("What do people say about the food quality?")

    def test_unrelated_question_with_answer_word_is_not_flagged(self) -> None:
        assert not _is_reply_status_question("What's the answer to improving my ratings?")


class TestReplyStatusAnswerConstant:
    def test_mentions_aio_dashboard_and_does_not_invent_a_number(self) -> None:
        assert "AIO dashboard" in REPLY_STATUS_ANSWER
        assert not any(char.isdigit() for char in REPLY_STATUS_ANSWER)
