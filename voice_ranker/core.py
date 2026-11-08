"""Audio preprocessing, pretrained speaker embeddings, ranking, and caching.

No model is trained by this project. On first inference, Hugging Face downloads
microsoft/wavlm-base-plus-sv; subsequent runs use its local cache.
"""
from __future__ import annotations

import csv
from contextlib import contextmanager, nullcontext
from dataclasses import InitVar, asdict, dataclass, fields
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import threading
from typing import Callable, Iterable, Iterator
from zipfile import BadZipFile, ZipFile, ZipInfo

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

SAMPLE_RATE = 16_000
MODEL_NAME = "microsoft/wavlm-base-plus-sv"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aiff", ".aif"}
CACHE_VERSION = "voice-ranker-v1"


class AudioError(RuntimeError):
    """An audio clip cannot provide reliable speech for a comparison."""


@dataclass
class ClipFeatures:
    embedding: np.ndarray
    duration_seconds: float
    speech_seconds: float
    n_segments: int
    warning: str = ""


@dataclass
class RankedClip:
    rank: int | None
    filename: str
    path: str
    similarity: float | None
    speech_seconds: float | None
    duration_seconds: float | None
    segments: int | None
    warning: str = ""
    error: str = ""
    archive_entry: InitVar[tuple[Path, ZipInfo] | None] = None

    def __post_init__(self, archive_entry: tuple[Path, ZipInfo] | None) -> None:
        self.archive_source = archive_entry


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-10:
        raise ValueError("Invalid or zero-length embedding")
    return arr / norm


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    a = normalize_embedding(first)
    b = normalize_embedding(second)
    if a.shape != b.shape:
        raise ValueError("Speaker embedding dimensions differ")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _run_ffmpeg(path: Path, *output_args: str) -> bytes:
    if not shutil.which("ffmpeg"):
        raise AudioError(
            f"Cannot decode {path.name}. Install FFmpeg for M4A/MP3/Opus, or convert to WAV."
        )
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), *output_args]
    try:
        process = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip()[-350:]
        raise AudioError(f"Could not decode {path.name}: {detail}") from exc
    return process.stdout


def _read_with_ffmpeg(path: Path) -> np.ndarray:
    data = _run_ffmpeg(path, "-f", "f32le", "-acodec", "pcm_f32le",
                       "-ac", "1", "-ar", str(SAMPLE_RATE), "-")
    return np.frombuffer(data, dtype="<f4").copy()


def read_audio(path: str | Path) -> np.ndarray:
    """Read a file as mono float32 at 16 kHz; never denoise or alter timbre."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"File not found: {path}")
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise AudioError(f"Unsupported file extension: {path.suffix}")
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sr <= 0 or len(audio) == 0:
            raise ValueError("empty or invalid audio")
        audio = np.mean(audio, axis=1)
        if sr != SAMPLE_RATE:
            divisor = math.gcd(sr, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor)
    except (RuntimeError, ValueError, OSError):
        audio = _read_with_ffmpeg(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise AudioError(f"Audio is empty: {path.name}")
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(np.abs(audio)) < 1e-4:
        raise AudioError(f"Audio is silent: {path.name}")
    return audio


def split_evenly(audio: np.ndarray, *, segment_seconds: float = 6.0,
                 max_segments: int = 6) -> list[np.ndarray]:
    """Uniformly sample speech across the clip, not just the opening seconds."""
    if not max_segments >= 1 or segment_seconds <= 0:
        raise ValueError("max_segments and segment_seconds must be positive")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or len(audio) == 0:
        raise AudioError("No speech samples to embed")
    window = max(1, int(round(segment_seconds * SAMPLE_RATE)))
    if len(audio) <= window:
        return [audio]
    n = min(max_segments, math.ceil(len(audio) / window))
    starts = np.rint(np.linspace(0, len(audio) - window, n)).astype(int)
    return [audio[start:start + window] for start in starts]


def find_candidates(items: Iterable[str | Path], reference: str | Path | None = None,
                    max_files: int = 5000, *, include_archives: bool = False) -> list[Path]:
    """Find audio files and optionally ZIP archives, scanning directories recursively."""
    if max_files <= 0:
        raise ValueError("max_files must be positive")
    extensions = AUDIO_EXTENSIONS | {".zip"} if include_archives else AUDIO_EXTENSIONS
    exclude = Path(reference).resolve() if reference is not None else None
    result: dict[str, Path] = {}
    audio_count = 0
    for item in items:
        if item is None or str(item).strip() == "":
            continue
        p = Path(item).expanduser()
        if p.is_dir():
            paths = (candidate for candidate in p.rglob("*") if candidate.is_file())
        elif p.is_file():
            paths = (p,)
        else:
            raise FileNotFoundError(f"Candidate path does not exist: {p}")
        for file in paths:
            if file.suffix.lower() not in extensions:
                continue
            resolved = file.resolve()
            if resolved == exclude:
                continue
            key = str(resolved).casefold()
            if key not in result and resolved.suffix.lower() != ".zip":
                audio_count += 1
            result[key] = resolved
            if audio_count > max_files:
                raise ValueError(f"More than {max_files} audio files. Narrow the folder or increase --max-files.")
    return sorted(result.values(), key=lambda p: str(p).casefold())


def _archive_audio(path: Path) -> tuple[ZipInfo, ...]:
    try:
        with ZipFile(path) as archive:
            return tuple(
                member for member in archive.infolist()
                if not member.is_dir() and PurePosixPath(member.filename).suffix.lower() in AUDIO_EXTENSIONS
            )
    except BadZipFile as exc:
        raise ValueError(f"Could not read ZIP archive: {path}") from exc


@contextmanager
def _candidate_file(path: Path, archive: ZipFile | None, member: ZipInfo | None) -> Iterator[Path]:
    if archive is None or member is None:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="voice-ranker-zip-") as folder:
        suffix = PurePosixPath(member.filename).suffix.lower()
        # Use a generated name so archive paths cannot affect where files are written.
        clip = Path(folder) / f"clip{suffix}"
        with archive.open(member) as stream, clip.open("wb") as output:
            shutil.copyfileobj(stream, output)
        yield clip


def _playable_wav(path: Path) -> bytes:
    if not path.is_file():
        raise AudioError(f"File not found: {path}")
    try:
        audio, sample_rate = sf.read(path, dtype="float32")
        with BytesIO() as output:
            sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
            return output.getvalue()
    except (RuntimeError, ValueError, OSError):
        with tempfile.TemporaryDirectory(prefix="voice-ranker-playback-") as folder:
            output_path = Path(folder) / "preview.wav"
            _run_ffmpeg(path, "-vn", "-c:a", "pcm_s16le", str(output_path))
            return output_path.read_bytes()


def preview_audio(clip: RankedClip) -> bytes:
    """Return browser-playable audio from the original clip or ZIP member."""
    if clip.archive_source is None:
        return _playable_wav(Path(clip.path))
    path, member = clip.archive_source
    with ZipFile(path) as archive, _candidate_file(path, archive, member) as source:
        return _playable_wav(source)


def save_csv(results: list[RankedClip], output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[column.name for column in fields(RankedClip)])
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))
    return output_path


def load_csv(report: str | Path, source: str | Path) -> list[RankedClip]:
    """Load an export and bind playable rows to an explicitly selected ZIP or folder."""
    root = Path(source).expanduser().resolve()
    is_folder = root.is_dir()
    if not is_folder and not (root.is_file() and root.suffix.lower() == ".zip"):
        raise ValueError("Choose the source ZIP or a local folder containing the original audio.")

    archives: dict[Path, dict[str, ZipInfo | None]] = {}
    archive_origin = None
    results = []
    with Path(report).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {column.name for column in fields(RankedClip)}.issubset(reader.fieldnames or []):
            raise ValueError("Choose a rankings CSV exported by Voice Similarity Ranker.")
        for row in reader:
            if not row["filename"] or not row["path"]:
                raise ValueError(f"Missing filename or source path on CSV line {reader.line_num}.")

            original, separator, member_name = row["path"].partition("!/")
            original_path = Path(original)
            local_path = None
            archive_entry = None
            if is_folder:
                if original_path.is_relative_to(root) and ".." not in original_path.parts:
                    resolved = original_path.resolve()
                    if resolved.is_relative_to(root) and resolved.is_file():
                        local_path = resolved
            elif separator and PurePosixPath(original.replace("\\", "/")).name.casefold() == root.name.casefold():
                if archive_origin is not None and archive_origin != original:
                    raise ValueError(f"The report contains multiple ZIPs named {root.name}. Choose their parent folder.")
                archive_origin = original
                local_path = root

            if separator and local_path and local_path.suffix.lower() == ".zip":
                if local_path not in archives:
                    members = {}
                    for member in _archive_audio(local_path):
                        # A CSV cannot distinguish duplicate members with identical names.
                        members[member.filename] = None if member.filename in members else member
                    archives[local_path] = members
                member = archives[local_path].get(member_name)
                if member is not None:
                    archive_entry = (local_path, member)

            playable_path = ""
            if archive_entry is not None:
                playable_path = f"{local_path}!/{member_name}"
            elif not separator and local_path is not None and local_path.suffix.lower() in AUDIO_EXTENSIONS:
                playable_path = str(local_path)
            results.append(RankedClip(
                int(row["rank"]) if row["rank"] else None,
                row["filename"],
                playable_path,
                float(row["similarity"]) if row["similarity"] else None,
                float(row["speech_seconds"]) if row["speech_seconds"] else None,
                float(row["duration_seconds"]) if row["duration_seconds"] else None,
                int(row["segments"]) if row["segments"] else None,
                row["warning"] or "",
                row["error"] or "",
                archive_entry=archive_entry,
            ))
    return results


class VoiceRanker:
    """Lazy-loaded WavLM speaker verifier with optional Silero VAD and disk cache."""

    def __init__(self, *, model_name: str = MODEL_NAME, device: str = "auto",
                 use_vad: bool = True, segment_seconds: float = 6.0,
                 max_segments: int = 6, cache_dir: str | Path | None = None,
                 use_cache: bool = True):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if max_segments < 1:
            raise ValueError("max_segments must be >= 1")
        self.model_name = model_name
        self.device_option = device
        self.use_vad = use_vad
        self.segment_seconds = segment_seconds
        self.max_segments = max_segments
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir or ".voice_ranker_cache").expanduser()
        self._extractor = None
        self._model = None
        self._vad = None
        self._device = None
        self._lock = threading.RLock()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        device = "cuda" if self.device_option == "auto" and torch.cuda.is_available() else self.device_option
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch can't see a CUDA GPU. Use --device cpu or install CUDA PyTorch.")
        try:
            extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            model = WavLMForXVector.from_pretrained(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                "Could not load the WavLM speaker model. Internet is needed on first run; "
                "check your connection, Hugging Face access and dependencies. "
                f"Original error: {exc}"
            ) from exc
        self._device = device
        self._model = model.to(device).eval()
        self._extractor = extractor

    def _load_vad(self) -> None:
        if not self.use_vad or self._vad is not None:
            return
        try:
            from silero_vad import load_silero_vad
            self._vad = load_silero_vad()
        except Exception as exc:
            raise RuntimeError(
                "Couldn't load Silero voice activity detection. Reinstall requirements, "
                "or run with --no-vad. " + str(exc)
            ) from exc

    def _speech_only(self, audio: np.ndarray) -> tuple[np.ndarray, float, str]:
        if not self.use_vad:
            return audio, len(audio) / SAMPLE_RATE, "VAD disabled; silence/music may affect the result"
        import torch
        from silero_vad import get_speech_timestamps
        self._load_vad()
        with torch.inference_mode():
            stamps = get_speech_timestamps(torch.from_numpy(audio), self._vad,
                                           sampling_rate=SAMPLE_RATE,
                                           min_speech_duration_ms=200,
                                           min_silence_duration_ms=150,
                                           speech_pad_ms=80)
        if not stamps:
            raise AudioError(
                "No speech detected. The file may be music/noise, or VAD may have missed a voice. "
                "Try turning VAD off to inspect the clip."
            )
        chunks = [audio[t["start"]:t["end"]] for t in stamps if t["end"] > t["start"]]
        if not chunks:
            raise AudioError("No usable speech detected")
        speech = np.concatenate(chunks)
        seconds = len(speech) / SAMPLE_RATE
        if seconds < 0.8:
            raise AudioError(f"Only {seconds:.2f}s of speech detected; needs at least 0.8s")
        warning = "" if seconds >= 2.0 else "Short speech sample: score may be unstable"
        return speech, seconds, warning

    def _embeddings(self, segments: list[np.ndarray]) -> np.ndarray:
        import torch
        self._load_model()
        outputs = []
        with torch.inference_mode():
            for clip in segments:
                # A single clip per inference avoids padding-induced artifacts.
                features = self._extractor(clip, sampling_rate=SAMPLE_RATE,
                                           return_tensors="pt")
                features = {key: tensor.to(self._device) for key, tensor in features.items()}
                emb = self._model(**features).embeddings[0].detach().float().cpu().numpy()
                outputs.append(normalize_embedding(emb))
        vectors = np.asarray(outputs, dtype=np.float32)
        if len(vectors) == 1:
            return vectors[0]
        prototype = normalize_embedding(vectors.mean(axis=0))
        # Discard at most one fifth of anomalous windows before pooling.
        keep = max(1, len(vectors) - len(vectors) // 5)
        top = np.argsort(vectors @ prototype)[-keep:]
        return normalize_embedding(vectors[top].mean(axis=0))

    def _cache_key(self, path: Path) -> str:
        stat = path.stat()
        values = {
            "v": CACHE_VERSION, "path": str(path.resolve()),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "model": self.model_name, "vad": self.use_vad,
            "segment_seconds": self.segment_seconds,
            "max_segments": self.max_segments,
        }
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def _get_cached(self, key: str) -> ClipFeatures | None:
        cache_file = self.cache_dir / f"{key}.npz"
        if not self.use_cache or not cache_file.exists():
            return None
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                return ClipFeatures(
                    embedding=normalize_embedding(data["embedding"]),
                    duration_seconds=float(data["duration_seconds"]),
                    speech_seconds=float(data["speech_seconds"]),
                    n_segments=int(data["n_segments"]),
                    warning=str(data["warning"]),
                )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _store_cache(self, key: str, feature: ClipFeatures) -> None:
        if not self.use_cache:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix=".tmp", suffix=".npz", dir=self.cache_dir)
            os.close(fd)
            try:
                np.savez_compressed(temp_name, embedding=feature.embedding,
                                    duration_seconds=feature.duration_seconds,
                                    speech_seconds=feature.speech_seconds,
                                    n_segments=feature.n_segments, warning=feature.warning)
                os.replace(temp_name, self.cache_dir / f"{key}.npz")
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
        except OSError:
            # A non-writable cache directory must never break scoring.
            return

    def features(self, path: str | Path, *, cache: bool = True) -> ClipFeatures:
        """Process one clip. Serialization protects the shared model and VAD."""
        with self._lock:
            path = Path(path).expanduser().resolve()
            if not path.is_file():
                raise AudioError(f"No such file: {path}")
            key = self._cache_key(path) if cache and self.use_cache else None
            if key is not None:
                cached = self._get_cached(key)
                if cached is not None:
                    return cached
            audio = read_audio(path)
            duration = len(audio) / SAMPLE_RATE
            speech, speech_sec, warning = self._speech_only(audio)
            if len(speech) / SAMPLE_RATE < 0.8:
                raise AudioError("Need at least 0.8 seconds of speech")
            segments = split_evenly(speech, segment_seconds=self.segment_seconds,
                                    max_segments=self.max_segments)
            embedding = self._embeddings(segments)
            result = ClipFeatures(embedding, duration, speech_sec, len(segments), warning)
            if key is not None:
                self._store_cache(key, result)
            return result

    def _score_candidate(self, path: Path, reference: ClipFeatures,
                         archive: ZipFile | None, member: ZipInfo | None) -> RankedClip:
        filename = f"{path.name}/{member.filename}" if member else path.name
        source = f"{path}!/{member.filename}" if member else str(path)
        try:
            with _candidate_file(path, archive, member) as clip:
                feat = self.features(clip, cache=False) if member else self.features(clip)
            score = cosine_similarity(reference.embedding, feat.embedding)
            return RankedClip(None, filename, source, score, round(feat.speech_seconds, 2),
                              round(feat.duration_seconds, 2), feat.n_segments, feat.warning,
                              archive_entry=(path, member) if member else None)
        except Exception as exc:
            return RankedClip(None, filename, source, None, None, None, None,
                              error=f"{type(exc).__name__}: {exc}")

    def rank(self, reference: str | Path,
             candidates: Iterable[str | Path],
             progress: Callable[[int, int, str], None] | None = None, *,
             max_files: int = 5000
             ) -> tuple[ClipFeatures, list[RankedClip]]:
        reference = Path(reference).expanduser().resolve()
        files = find_candidates(candidates, reference=reference, max_files=max_files,
                                include_archives=True)
        batches: list[tuple[Path, tuple[ZipInfo | None, ...]]] = []
        total = 0
        for path in files:
            members = _archive_audio(path) if path.suffix.lower() == ".zip" else (None,)
            if members:
                batches.append((path, members))
                total += len(members)
            if total > max_files:
                raise ValueError(f"More than {max_files} audio files. Narrow the input or increase --max-files.")
        if not total:
            raise ValueError("No candidate audio files found. Select audio files, a folder, or a ZIP archive.")
        self._load_model()
        reference_features = self.features(reference)
        ranked: list[RankedClip] = []
        failed: list[RankedClip] = []
        index = 0
        for path, members in batches:
            with (ZipFile(path) if members[0] is not None else nullcontext()) as archive:
                for member in members:
                    item = self._score_candidate(path, reference_features, archive, member)
                    (failed if item.error else ranked).append(item)
                    index += 1
                    if progress is not None:
                        progress(index, total, item.filename)
        ranked.sort(key=lambda row: row.similarity if row.similarity is not None else -100,
                    reverse=True)
        for idx, row in enumerate(ranked, start=1):
            row.rank = idx
        return reference_features, ranked + failed
