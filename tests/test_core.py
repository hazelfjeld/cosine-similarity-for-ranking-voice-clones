from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pytest
import soundfile as sf

from voice_ranker.core import (
    AUDIO_EXTENSIONS, AudioError, ClipFeatures, RankedClip, VoiceRanker,
    cosine_similarity, find_candidates, load_csv, preview_audio, read_audio, save_csv,
    split_evenly, SAMPLE_RATE,
)


def _write_tone(path: Path, freq: int = 200, sample_rate: int = 22050, seconds: float = 2.0):
    t = np.arange(round(sample_rate * seconds)) / sample_rate
    sf.write(path, 0.3 * np.sin(2 * np.pi * freq * t), sample_rate)


def test_read_audio_resamples_stereo(tmp_path):
    file = tmp_path / 'stereo.wav'
    t = np.arange(22050) / 22050
    sf.write(file, np.stack([0.3 * np.sin(2*np.pi*220*t), 0.3 * np.sin(2*np.pi*330*t)], axis=-1), 22050)
    out = read_audio(file)
    assert out.ndim == 1
    assert abs(len(out) - 16000) <= 2
    assert np.max(np.abs(out)) > 0.05


def test_silence_rejected(tmp_path):
    file = tmp_path / 'silence.wav'
    sf.write(file, np.zeros(16000), 16000)
    with pytest.raises(AudioError, match='silent'):
        read_audio(file)


def test_cosine():
    assert cosine_similarity([3, 0], [0.5, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 2]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 0])


def test_even_sampling():
    audio = np.arange(20 * SAMPLE_RATE, dtype=np.float32)
    chunks = split_evenly(audio, segment_seconds=6, max_segments=3)
    assert len(chunks) == 3
    assert len(chunks[0]) == 6 * SAMPLE_RATE
    assert chunks[0][0] == 0
    assert chunks[-1][-1] == audio[-1]
    assert len(split_evenly(np.ones(100, np.float32))) == 1


def test_find_candidates_excludes_reference_deduplicates(tmp_path):
    a = tmp_path / 'original.wav'
    sub = tmp_path / 'sub'
    sub.mkdir()
    b = sub / 'out.wav'
    a.write_bytes(b'anything')
    b.write_bytes(b'anything')
    (sub / 'notes.txt').write_text('ignore')
    found = find_candidates([tmp_path, b], reference=a)
    assert found == [b.resolve()]


def test_cache_roundtrip_and_mutation(tmp_path, monkeypatch):
    path = tmp_path / 'voice.wav'
    _write_tone(path)
    ranker = VoiceRanker(use_vad=False, cache_dir=tmp_path / 'cache')
    calls = []
    def fake_embs(segments):
        calls.append(1)
        return np.asarray([1., 0.], dtype=np.float32)
    monkeypatch.setattr(ranker, '_embeddings', fake_embs)
    a = ranker.features(path)
    b = ranker.features(path)
    assert len(calls) == 1
    assert np.array_equal(a.embedding, b.embedding)
    assert (tmp_path / 'cache').exists()
    _write_tone(path, freq=440, seconds=3)
    c = ranker.features(path)
    assert len(calls) == 2
    assert c.duration_seconds > a.duration_seconds


def test_rank_sorted_with_failed_file(tmp_path, monkeypatch):
    reference = tmp_path / 'reference.wav'
    first = tmp_path / 'first.wav'
    second = tmp_path / 'second.wav'
    broken = tmp_path / 'broken.wav'
    for f in (reference, first, second, broken):
        f.write_bytes(b'fake')
    ranker = VoiceRanker(use_cache=False)
    monkeypatch.setattr(ranker, '_load_model', lambda: None)
    def fake_feat(path):
        name = Path(path).stem
        if name == 'broken':
            raise AudioError('corrupt file')
        embedding = {'reference': [1.,0.], 'first': [0.4,0.6], 'second': [0.9,0.1]}[name]
        return ClipFeatures(np.array(embedding, dtype=np.float32), 2.0, 2.0, 1)
    monkeypatch.setattr(ranker, 'features', fake_feat)
    _, rows = ranker.rank(reference, [first, second, broken])
    assert [r.filename for r in rows] == ['second.wav', 'first.wav', 'broken.wav']
    assert [r.rank for r in rows] == [1, 2, None]
    assert rows[-1].error
    csv_file = save_csv(rows, tmp_path / 'out' / 'results.csv')
    text = csv_file.read_text(encoding='utf-8-sig')
    assert 'second.wav' in text and 'corrupt file' in text


def test_rank_zip_audio_in_nested_folders(tmp_path, monkeypatch):
    reference = tmp_path / 'reference.wav'
    _write_tone(reference)
    folder = tmp_path / 'candidates' / 'nested'
    folder.mkdir(parents=True)
    archive_path = folder / 'samples.ZIP'
    with ZipFile(archive_path, 'w') as archive:
        archive.write(reference, 'first/deep/clip.wav')
        archive.write(reference, 'second/clip.WAV')
        archive.writestr('second/notes.txt', 'ignore')
    loose = folder / 'loose.wav'
    _write_tone(loose)

    ranker = VoiceRanker(use_vad=False, cache_dir=tmp_path / 'cache')
    monkeypatch.setattr(ranker, '_load_model', lambda: None)
    monkeypatch.setattr(ranker, '_embeddings', lambda segments: np.array([1., 0.]))
    progress = []
    _, rows = ranker.rank(reference, [folder.parent, archive_path],
                          progress=lambda i, n, name: progress.append((i, n)),
                          max_files=3)

    assert [row.filename for row in rows] == [
        'loose.wav', 'samples.ZIP/first/deep/clip.wav', 'samples.ZIP/second/clip.WAV'
    ]
    assert [row.rank for row in rows] == [1, 2, 3]
    assert rows[1].path == f'{archive_path}!/first/deep/clip.wav'
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert len(list((tmp_path / 'cache').glob('*.npz'))) == 2
    with pytest.raises(ValueError, match='More than 2 audio files'):
        ranker.rank(reference, [folder.parent], max_files=2)


def test_preview_audio_preserves_full_clip_from_file_and_zip(tmp_path, monkeypatch):
    reference = tmp_path / 'reference.wav'
    _write_tone(reference)
    source = tmp_path / 'stereo.aiff'
    sample_rate = 44100
    stereo = np.tile([0.125, -0.25], (sample_rate, 1))
    sf.write(source, stereo, sample_rate)
    archive_path = tmp_path / 'samples.zip'
    with ZipFile(archive_path, 'w') as archive:
        archive.write(source, 'batch/nested/stereo.aiff')

    ranker = VoiceRanker(use_vad=False, use_cache=False)
    monkeypatch.setattr(ranker, '_load_model', lambda: None)
    monkeypatch.setattr(ranker, '_embeddings', lambda segments: np.array([1., 0.]))
    _, rows = ranker.rank(reference, [source, archive_path])

    assert len(rows) == 2 and all(row.rank is not None for row in rows)
    for row in rows:
        audio, rate = sf.read(BytesIO(preview_audio(row)))
        assert rate == sample_rate
        assert np.array_equal(audio, stereo)


def test_ui_constructs():
    from voice_ranker.ui import make_app
    app = make_app()
    assert app is not None
    app.close()


def test_saved_rankings_preview_unscored_zip_member(tmp_path):
    import gradio as gr
    from voice_ranker.ui import open_saved_rankings, play_table_clip

    source = tmp_path / "voice.wav"
    _write_tone(source)
    archive_path = tmp_path / "takes.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.write(source, "batch/unscored.wav")

    previous_path = tmp_path / "previous_location" / "takes.zip"
    report = save_csv([
        RankedClip(1, "missing.wav", f"{previous_path}!/batch/missing.wav", 0.95, 2.0, 2.0, 1),
        RankedClip(None, "unscored.wav", f"{previous_path}!/batch/unscored.wav",
                   None, None, None, None, error="No speech detected"),
    ], tmp_path / "rankings.csv")

    table, _, _, clips, picker, _ = open_saved_rankings(str(report), str(archive_path))
    assert table[0][0] == "Unavailable"
    assert table[1][0] == "▶ Play"
    assert table[1][-1] == "No speech detected"
    assert len(picker["choices"]) == 1
    event = gr.SelectData(None, {"index": [1, 0], "value": table[1][0], "row_value": table[1]})
    audio, selected = play_table_clip(clips, event)
    assert selected["value"] == picker["choices"][0][1]
    assert sf.info(BytesIO(audio)).duration == pytest.approx(2.0)


def test_saved_rankings_folder_only_uses_sources_inside_selected_folder(tmp_path):
    folder = tmp_path / "audio"
    folder.mkdir()
    inside = folder / "inside.wav"
    outside = tmp_path / "outside.wav"
    _write_tone(inside)
    _write_tone(outside)
    report = save_csv([
        RankedClip(1, path.name, str(path), 0.9, 2.0, 2.0, 1) for path in (inside, outside)
    ], tmp_path / "rankings.csv")

    assert [row.path for row in load_csv(report, folder)] == [str(inside), ""]
