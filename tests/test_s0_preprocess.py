"""Tests for stage S0, against ffmpeg-generated clips and the real corpus."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from avannotate.ffmpeg import (
    FFmpegError,
    choose_encoder,
    detect_cuts,
    encoder_names,
    probe_media,
    scene_scores,
    shots_from_cuts,
)
from avannotate.stages import s0_preprocess
from avannotate.stages.base import StageContext


def _context(source: Path, root: Path, **config: object) -> StageContext:
    return StageContext(
        video_id=source.stem,
        source=source,
        work_dir=root / "work" / source.stem,
        config=config,
    )


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #


def test_shots_from_cuts_tiles_the_video() -> None:
    shots = shots_from_cuts((2.0, 5.0), 8.0)
    assert shots == ((0.0, 2.0), (2.0, 5.0), (5.0, 8.0))


def test_shots_from_cuts_with_no_cuts_is_one_shot() -> None:
    assert shots_from_cuts((), 4.0) == ((0.0, 4.0),)


def test_shots_from_cuts_ignores_out_of_range_and_unsorted_cuts() -> None:
    """A detector can report a cut at 0.0, past the end, or out of order."""

    assert shots_from_cuts((0.0, 10.0, -1.0), 4.0) == ((0.0, 4.0),)
    assert shots_from_cuts((3.0, 1.0), 4.0) == ((0.0, 1.0), (1.0, 3.0), (3.0, 4.0))


def test_shots_from_cuts_rejects_a_nonpositive_duration() -> None:
    with pytest.raises(ValueError):
        shots_from_cuts((), 0.0)


# --------------------------------------------------------------------------- #
# ffmpeg wrappers
# --------------------------------------------------------------------------- #


def test_probe_reads_the_container(single_shot_video: Path) -> None:
    info = probe_media(single_shot_video)
    assert info.width == 320
    assert info.height == 240
    assert info.fps == pytest.approx(25.0)
    assert info.duration == pytest.approx(3.0, abs=0.1)
    assert info.has_audio
    assert info.sample_rate == 16000


def test_probe_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FFmpegError):
        probe_media(tmp_path / "absent.mp4")


def test_detect_cuts_finds_a_hard_cut(two_shot_video: Path) -> None:
    cuts = detect_cuts(two_shot_video, threshold=0.3)
    assert len(cuts) == 1
    assert cuts[0] == pytest.approx(2.0, abs=0.15)


def test_detect_cuts_finds_nothing_in_a_continuous_shot(single_shot_video: Path) -> None:
    assert detect_cuts(single_shot_video, threshold=0.3) == ()


def test_scene_scores_distinguishes_no_cuts_from_a_dead_filter(
    single_shot_video: Path, two_shot_video: Path
) -> None:
    """The diagnostic exists because those two look identical from detect_cuts."""

    flat = scene_scores(single_shot_video)
    cut = scene_scores(two_shot_video)

    assert len(flat) > 50
    assert max(flat) < 0.3
    assert max(cut) > 0.3


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #


def test_run_writes_probe_audio_and_shots(single_shot_video: Path, tmp_path: Path) -> None:
    context = _context(single_shot_video, tmp_path)
    result = s0_preprocess.run(context)

    assert not result.skipped
    assert (context.work_dir / "s0-preprocess" / "probe.json").is_file()
    assert (context.work_dir / "s0-preprocess" / "shots.json").is_file()
    assert (context.work_dir / "s0-preprocess" / "mix.wav").is_file()
    assert result.summary["shots"] == 1


def test_audio_is_resampled_to_16k_mono(single_shot_video: Path, tmp_path: Path) -> None:
    context = _context(single_shot_video, tmp_path)
    s0_preprocess.run(context)

    with wave.open(str(s0_preprocess.audio_path(context))) as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2  # PCM16


def test_a_silent_video_is_reported_not_half_processed(
    video_without_audio: Path, tmp_path: Path
) -> None:
    """No audio means S0 cannot do its job; the batch driver needs to hear that."""

    context = _context(video_without_audio, tmp_path)
    with pytest.raises(ValueError, match="no audio track"):
        s0_preprocess.run(context)


def test_second_run_skips(single_shot_video: Path, tmp_path: Path) -> None:
    context = _context(single_shot_video, tmp_path)
    assert not s0_preprocess.run(context).skipped

    second = s0_preprocess.run(context)
    assert second.skipped
    assert "unchanged" in second.reason


def test_force_reruns(single_shot_video: Path, tmp_path: Path) -> None:
    context = _context(single_shot_video, tmp_path)
    s0_preprocess.run(context)
    assert not s0_preprocess.run(context, force=True).skipped


def test_a_config_change_invalidates_the_cache(single_shot_video: Path, tmp_path: Path) -> None:
    s0_preprocess.run(_context(single_shot_video, tmp_path))
    changed = s0_preprocess.run(_context(single_shot_video, tmp_path, cut_threshold=0.05))
    assert not changed.skipped
    assert "inputs changed" in changed.reason


def test_a_tampered_artifact_invalidates_the_cache(
    single_shot_video: Path, tmp_path: Path
) -> None:
    context = _context(single_shot_video, tmp_path)
    s0_preprocess.run(context)

    with s0_preprocess.audio_path(context).open("ab") as handle:
        handle.write(b"\x00" * 32)

    rerun = s0_preprocess.run(context)
    assert not rerun.skipped
    assert "artifact" in rerun.reason


def test_video_duration_is_authoritative_over_the_audio_track(
    single_shot_video: Path, tmp_path: Path
) -> None:
    """AAC padding makes the demuxed track longer; the video must win, and the
    surplus must be recorded rather than hidden."""

    context = _context(single_shot_video, tmp_path)
    s0_preprocess.run(context)
    timeline = s0_preprocess.load_timeline(context)

    info = probe_media(single_shot_video)
    assert timeline.duration == pytest.approx(info.duration)
    assert timeline.audio_duration >= timeline.duration
    assert timeline.audio_delta == pytest.approx(
        timeline.audio_duration - timeline.duration
    )


def test_loaders_round_trip_what_run_wrote(two_shot_video: Path, tmp_path: Path) -> None:
    context = _context(two_shot_video, tmp_path)
    s0_preprocess.run(context)

    shots = s0_preprocess.load_shots(context)
    assert len(shots) == 2
    assert shots[0][0] == 1  # 1-based, to match the script's [SHOT n ...]
    assert shots[0][1] == 0.0
    assert shots[-1][2] == pytest.approx(s0_preprocess.load_timeline(context).duration)


def test_loaders_fail_loudly_when_the_stage_has_not_run(tmp_path: Path) -> None:
    context = StageContext(
        video_id="x", source=tmp_path / "x.mp4", work_dir=tmp_path / "work" / "x"
    )
    with pytest.raises(FileNotFoundError, match="run s0-preprocess first"):
        s0_preprocess.load_timeline(context)
    with pytest.raises(FileNotFoundError, match="run s0-preprocess first"):
        s0_preprocess.load_shots(context)
    with pytest.raises(FileNotFoundError, match="run s0-preprocess first"):
        s0_preprocess.audio_path(context)


# --------------------------------------------------------------------------- #
# the real corpus
# --------------------------------------------------------------------------- #


def test_sample_corpus_probes_and_demuxes(sample_videos: tuple[Path, ...], tmp_path: Path) -> None:
    for source in sample_videos:
        context = _context(source, tmp_path)
        result = s0_preprocess.run(context)
        assert not result.skipped

        timeline = s0_preprocess.load_timeline(context)
        assert timeline.duration > 0.0
        assert timeline.fps > 0.0
        assert timeline.frame_count > 0
        # Every sample is already 16 kHz mono, so the delta is pure AAC padding
        # and must stay small.  A large one would mean the demux went wrong.
        assert 0.0 <= timeline.audio_delta < 0.2


# --------------------------------------------------------------------------- #
# which video encoder the cropped clips get
# --------------------------------------------------------------------------- #


def test_encoder_names_reads_a_listing_but_not_its_headings() -> None:
    """``-encoders`` output has headings, blanks, and one line per encoder."""

    listing = (
        "Encoders:\n"
        " V..... = Video\n"
        " ------\n"
        " V....D libx264              H.264 / AVC (codec h264)\n"
        " V....D libx264rgb           H.264 / AVC (codec h264)\n"
        " A....D aac                  AAC (Advanced Audio Coding)\n"
    )
    names = encoder_names(listing)
    assert "libx264" in names
    assert "aac" in names
    # A different encoder with a longer name must not stand in for the shorter
    # one -- substring matching here would silently pick the wrong codec.
    assert "libx264rgb" in names


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        ({"libx264", "mpeg4"}, "libx264"),
        # A --disable-gpl build: no x264, no x265, and openh264 instead.
        ({"libopenh264", "h264_nvenc", "mpeg4"}, "libopenh264"),
        ({"h264_nvenc", "mpeg4"}, "h264_nvenc"),
        ({"mpeg4"}, "mpeg4"),
    ],
)
def test_choose_encoder_takes_the_best_one_present(
    available: set[str], expected: str
) -> None:
    assert choose_encoder(available) == expected


def test_choose_encoder_says_so_when_there_is_nothing_to_choose() -> None:
    with pytest.raises(FFmpegError, match="none of"):
        choose_encoder({"libvpx", "libaom-av1"})
