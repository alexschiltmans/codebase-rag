"""Unit tests for data_ingestion/truncation.py."""

from langchain_core.documents import Document

from codebase_rag.data_ingestion.truncation import format_truncation_report, measure_truncation


def _doc(file_type: str, text: str) -> Document:
    return Document(page_content=text, metadata={"file_type": file_type, "source": f"a{file_type}"})


def _count_words(texts: list[str]) -> list[int]:
    return [len(text.split()) for text in texts]


class TestMeasureTruncation:
    """Tests for measure_truncation."""

    def test_counts_over_limit_chunks_overall(self) -> None:
        docs = [_doc(".py", "w " * 20), _doc(".py", "w " * 4)]

        report = measure_truncation(docs, _count_words, limit=10)

        assert (report.chunks, report.over_limit, report.limit) == (2, 1, 10)
        assert report.share == 50.0

    def test_reports_share_per_file_type(self) -> None:
        docs = [
            _doc(".json", "w " * 20),
            _doc(".json", "w " * 20),
            _doc(".py", "w " * 20),
            _doc(".py", "w " * 2),
            _doc(".py", "w " * 2),
            _doc(".py", "w " * 2),
        ]

        report = measure_truncation(docs, _count_words, limit=10)

        shares = {entry.file_type: entry.share for entry in report.by_type}
        assert shares == {".json": 100.0, ".py": 25.0}

    def test_worst_type_is_reported_first(self) -> None:
        docs = [_doc(".py", "w " * 2), _doc(".py", "w " * 20), _doc(".json", "w " * 20)]

        report = measure_truncation(docs, _count_words, limit=10)

        assert [entry.file_type for entry in report.by_type] == [".json", ".py"]

    def test_a_clean_corpus_reports_zero(self) -> None:
        report = measure_truncation([_doc(".py", "w w")], _count_words, limit=10)

        assert report.over_limit == 0
        assert report.by_type[0].share == 0.0

    def test_counts_every_chunk_exactly_once(self) -> None:
        seen: list[str] = []

        def counting(texts: list[str]) -> list[int]:
            seen.extend(texts)
            return _count_words(texts)

        docs = [_doc(".py", f"chunk {i}") for i in range(25)]
        measure_truncation(docs, counting, limit=10, batch_size=10)

        assert seen == [doc.page_content for doc in docs]

    def test_tokenizes_in_batches_rather_than_one_call(self) -> None:
        sizes: list[int] = []

        def counting(texts: list[str]) -> list[int]:
            sizes.append(len(texts))
            return _count_words(texts)

        measure_truncation([_doc(".py", "w") for _ in range(25)], counting, limit=10, batch_size=10)

        assert sizes == [10, 10, 5]

    def test_chunks_without_file_type_fall_back_to_the_source_suffix(self) -> None:
        doc = Document(page_content="w " * 20, metadata={"source": "/repo/x.svg"})

        report = measure_truncation([doc], _count_words, limit=10)

        assert report.by_type[0].file_type == ".svg"

    def test_empty_run_does_not_call_the_tokenizer(self) -> None:
        def exploding(texts: list[str]) -> list[int]:
            raise AssertionError("tokenizer called for an empty corpus")

        assert measure_truncation([], exploding, limit=10).chunks == 0

    def test_empty_run_measures_nothing(self) -> None:
        report = measure_truncation([], _count_words, limit=10)

        assert (report.chunks, report.over_limit, report.by_type) == (0, 0, ())
        assert report.share == 0.0


class TestFormatTruncationReport:
    """Tests for format_truncation_report."""

    def test_truncation_is_reported_with_the_limit_and_the_types(self) -> None:
        docs = [_doc(".json", "w " * 20), _doc(".py", "w w")]

        lines = format_truncation_report(measure_truncation(docs, _count_words, limit=10))

        assert "1 of 2 chunks (50.00%)" in lines[0]
        assert "10-token" in lines[0]
        assert lines[1].strip() == ".json: 1/1 (100.00%)"

    def test_clean_types_are_left_out_of_the_breakdown(self) -> None:
        docs = [_doc(".json", "w " * 20), _doc(".py", "w w")]

        lines = format_truncation_report(measure_truncation(docs, _count_words, limit=10))

        assert not any(".py" in line for line in lines)

    def test_a_clean_run_says_so_rather_than_staying_silent(self) -> None:
        lines = format_truncation_report(measure_truncation([_doc(".py", "w w")], _count_words, limit=10))

        assert lines == ["Truncation check: 0 of 1 chunks exceed the 10-token embedding limit, nothing was truncated"]

    def test_an_empty_run_does_not_claim_a_clean_bill(self) -> None:
        lines = format_truncation_report(measure_truncation([], _count_words, limit=10))

        assert lines == ["Truncation check: nothing to index, no chunks measured"]
