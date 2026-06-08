from audio.types.audio_quality_result import AudioQualityResult
from report.utils.summarize_comparison import summarize_comparison


def make_result(name: str, snr_db: float) -> AudioQualityResult:
    return AudioQualityResult(
        name=name,
        duration_seconds=60.0,
        silence_gap_snr_db=snr_db,
        speech_fraction=0.2,
    )


class TestSummarizeComparison:
    def test_names_the_higher_snr_recording(self) -> None:
        verdict = summarize_comparison([make_result("a", 10.0), make_result("b", 13.0)])
        assert "b has the higher silence-gap SNR" in verdict
        assert "3.0 dB" in verdict

    def test_calls_a_small_difference_comparable(self) -> None:
        verdict = summarize_comparison([make_result("a", 10.0), make_result("b", 10.2)])
        assert "comparable" in verdict

    def test_ranks_best_and_worst_across_many(self) -> None:
        verdict = summarize_comparison(
            [make_result("a", 10.0), make_result("b", 13.0), make_result("c", 7.0)],
        )
        assert "b has the highest silence-gap SNR" in verdict
        assert "c the lowest" in verdict
