"""Local Gradio upload-and-rank browser UI."""
from __future__ import annotations

from pathlib import Path
import tempfile
from uuid import uuid4

import gradio as gr

from .core import AUDIO_EXTENSIONS, RankedClip, VoiceRanker, load_csv, preview_audio, save_csv

# Reuse the pretrained model across browser requests (and disk-cache embeddings).
_rankers: dict[tuple[bool, str], VoiceRanker] = {}


def _get_ranker(use_vad: bool, device: str) -> VoiceRanker:
    key = (use_vad, device)
    if key not in _rankers:
        _rankers[key] = VoiceRanker(use_vad=use_vad, device=device)
    return _rankers[key]


def run_ranking(reference: str | None, uploaded_files: list[str] | None,
                local_folder: str, use_vad: bool, device: str,
                progress=gr.Progress()):
    if not reference:
        raise gr.Error("First upload a reference voice sample.")
    candidates = list(uploaded_files or [])
    if local_folder and local_folder.strip():
        candidates.append(local_folder.strip().strip('"'))
    if not candidates:
        raise gr.Error("Upload candidate clips or a ZIP archive, or enter a local path.")
    try:
        ranker = _get_ranker(use_vad, device)
        progress(0, desc="Finding audio and preparing reference")
        ref_features, ranked = ranker.rank(
            reference, candidates, progress=lambda i, n, name: progress(i / n, desc=f"Comparing {i}/{n}: {name[-40:]}")
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    tmp = Path(tempfile.mkdtemp(prefix="voice-ranker-")) / "rankings.csv"
    save_csv(ranked, tmp)
    valid = sum(not row.error for row in ranked)
    summary = (
        f"**Finished:** {valid}/{len(ranked)} clips ranked. "
        f"Reference speech: **{ref_features.speech_seconds:.1f}s** "
        f"({ref_features.n_segments} analysis windows). "
        f"Top match: **{next((r.filename for r in ranked if not r.error), 'none')}**.\n\n"
        "Scores are cosine similarities, **not confidence percentages**. "
        "Higher means the speaker embedding is closer to the reference."
    )
    return _result_outputs(ranked, str(tmp), summary)


def open_saved_rankings(report: str | None, source: str):
    if not report or not source.strip():
        raise gr.Error("Choose a rankings CSV and enter the path to its source ZIP or audio folder.")
    try:
        ranked = load_csv(report, source.strip().strip('"'))
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    available = sum(bool(clip.path) for clip in ranked)
    summary = f"**Saved rankings:** {available}/{len(ranked)} clips available to play."
    return _result_outputs(ranked, report, summary)


def _result_outputs(ranked: list[RankedClip], report: str, summary: str):
    result_id = uuid4().hex
    clips = {f"{result_id}:{index}": item for index, item in enumerate(ranked)}
    table = []
    choices = []
    for key, item in clips.items():
        table.append(["▶ Play" if item.path else "Unavailable",
                      item.rank if item.rank is not None else "—",
                      item.filename,
                      round(item.similarity, 4) if item.similarity is not None else "—",
                      item.speech_seconds if item.speech_seconds is not None else "—",
                      item.warning or item.error])
        if item.path:
            label = f"#{item.rank}" if item.rank is not None else "Unranked"
            choices.append((f"{label} · {item.filename}", key))
    picker = gr.update(choices=choices, value=None, interactive=bool(choices))
    return table, report, summary, clips, picker, None


def load_ranked_clip(selection: str | None, clips: dict[str, RankedClip]) -> bytes | None:
    clip = clips.get(selection) if selection is not None else None
    if clip is None or not clip.path:
        return None
    try:
        return preview_audio(clip)
    except Exception as exc:
        gr.Warning(f"Could not load {clip.filename}: {exc}")
        return None


def play_table_clip(clips: dict[str, RankedClip], event: gr.SelectData):
    if not event.index or event.index[1] != 0 or event.value != "▶ Play":
        return gr.skip(), gr.skip()
    index = event.index[0]
    if not 0 <= index < len(clips):
        return gr.skip(), gr.skip()
    key, clip = list(clips.items())[index]
    if not event.row_value or event.row_value[2] != clip.filename:
        return gr.skip(), gr.skip()
    return load_ranked_clip(key, clips), gr.update(value=key)


def make_app() -> gr.Blocks:
    with gr.Blocks(title="Voice Similarity Ranker") as demo:
        gr.Markdown(
            "# Voice Similarity Ranker\n"
            "**Reference voice → generated samples → sorted voice-identity similarities.** "
            "Runs locally. No paid API or training required."
        )
        with gr.Row():
            reference = gr.Audio(label="1 · Reference voice clip", sources=["upload"], type="filepath")
            with gr.Column():
                candidates = gr.File(label="2 · Candidate audio files or ZIP archives",
                                     file_count="multiple", type="filepath",
                                     file_types=sorted(AUDIO_EXTENSIONS | {".zip"}))
                folder = gr.Textbox(label="OR local folder or ZIP path",
                                    placeholder=r"C:\voices\outputs  (includes audio in subfolders)")
        with gr.Accordion("Options", open=False):
            vad = gr.Checkbox(value=True, label="Use voice activity detection to discard non-speech sections")
            device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="auto",
                                 label="Compute device (auto uses GPU when available)")
        run = gr.Button("Rank voice samples", variant="primary")
        with gr.Accordion("Open saved rankings", open=False):
            saved_report = gr.File(label="Rankings CSV", type="filepath", file_types=[".csv"])
            saved_source = gr.Textbox(label="Local path to the source ZIP or audio folder",
                                      placeholder=r"C:\voices\takes.zip")
            open_saved = gr.Button("Load saved rankings")
        status = gr.Markdown("Results will appear here.")
        clips = gr.State({})
        selection = gr.Dropdown(label="Choose a clip to listen to", choices=[],
                                value=None, interactive=False)
        player = gr.Audio(label="Clip playback", interactive=False, autoplay=True)
        table = gr.Dataframe(headers=["Listen", "Rank", "File", "Cosine similarity", "Speech seconds", "Notes"],
                             datatype="str", interactive=False, label="Results · click ▶ Play to listen")
        report = gr.File(label="Download ranked CSV")
        gr.Markdown(
            "**What this measures:** perceived speaker identity, mainly timbre/acoustic voice patterns. "
            "It does **not** directly measure emotion, acting, pronunciation, music bleed, or audio glitches. "
            "Speech under background music can still distort scores. "
            "First launch downloads an open pretrained model. Later launches use local files."
        )
        results = [table, report, status, clips, selection, player]
        run.click(fn=run_ranking, inputs=[reference, candidates, folder, vad, device],
                  outputs=results, concurrency_id="ranking")
        open_saved.click(fn=open_saved_rankings, inputs=[saved_report, saved_source],
                         outputs=results, concurrency_id="ranking")
        selection.input(fn=load_ranked_clip, inputs=[selection, clips], outputs=player,
                        concurrency_id="ranking", trigger_mode="always_last", show_progress="minimal")
        table.select(fn=play_table_clip, inputs=clips, outputs=[player, selection],
                     concurrency_id="ranking", trigger_mode="always_last", show_progress="minimal")
    return demo
