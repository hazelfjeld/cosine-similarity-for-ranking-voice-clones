# Voice Similarity Ranker

A **free, local** voice-identity similarity ranker. Give it one reference voice clip and any number of generated clips. It automatically orders the generated clips from most to least similar to the speaker in the reference, with CSV/JSON exports for later curation.

**Model:** [`microsoft/wavlm-base-plus-sv`](https://huggingface.co/microsoft/wavlm-base-plus-sv), a pretrained WavLM speaker-verification model. **Optional speech detection:** [Silero VAD](https://github.com/snakers4/silero-vad). No new model training, Adaption credits, API keys, or cloud upload of your audio required.

## Windows: the easiest route

1. Install **Python 3.11 or 3.12 (64-bit)** from python.org if you do not have it. Check "Add Python to PATH". For MP3/M4A support on older systems, install [FFmpeg](https://ffmpeg.org/download.html); ordinary WAV files need no FFmpeg.
2. Unzip this project into a writable folder, then double-click **`start_windows.bat`**.
3. The first run installs Python dependencies and, when you first press Rank, downloads the pretrained WavLM checkpoint. This needs internet and can be a substantial download. After that it works offline from local cache.
4. In the browser, upload your **reference** (20–30 seconds is plenty). Then select candidate audio files or ZIP archives, or paste the path to a local folder or ZIP. Click **Rank voice samples**.
5. Click **▶ Play** beside a result or use **Choose a clip to listen to**. If playback doesn't start automatically, press Play in the audio player. Clips without a score are included for listening.

To listen to a previous run, expand **Open saved rankings**, upload its `rankings.csv`, and enter the local path to the source ZIP or a folder containing the original audio. Click **Load saved rankings**. This reads the saved scores and loads audio on demand. Keep the source audio alongside the report for later listening.

The ranker searches subfolders in local folders and ZIP archives. Results for archived clips show the ZIP name and the path inside it, so clips with the same filename in different subfolders are distinguishable. For large datasets, paste the local path to avoid uploading the files through the browser. All scores and cached embeddings stay on your own machine. The Gradio server listens only on `127.0.0.1`, not the internet.

> **If Windows blocks the .bat:** Open PowerShell in the unzipped folder and run `python -m venv .venv`, `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `.\.venv\Scripts\python.exe app.py`.

## Command line (automated generation pipeline)

After installation, run:

```powershell
.\.venv\Scripts\python.exe -m voice_ranker.cli --reference "C:\clips\reference.wav" --candidates "C:\clips\generated" --output "C:\clips\rankings.csv" --json "C:\clips\rankings.json"
```

Or on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m voice_ranker.cli --reference reference.wav --candidates generated/ --output rankings.csv --json rankings.json
```

Pass audio files, folders, or ZIP archives to `--candidates`. You can mix input types in one run.

**Options:** `--no-vad` bypasses speech detection (useful if a noisy reference is being rejected); `--device cpu` forces CPU; `--device cuda` requires an NVIDIA CUDA-enabled PyTorch; `--max-segments 6` controls the number of windows per clip; `--no-cache` forces fresh analysis. Inference automatically uses the CUDA GPU if PyTorch can see it, otherwise CPU.

## How rankings are computed

1. Decode `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.opus`, or `.aiff`, downmix to mono, and resample to **16 kHz**.
2. Detect voice-active intervals using Silero VAD and discard non-speech sections. This removes leading/trailing silent gaps and many music-only sections; **it cannot remove music playing underneath a voice**. Use `--no-vad` for comparison if it misdetects voice.
3. Sample up to six **6-second** windows spread across each clip, rather than cherry-picking the first seconds.
4. Get a  speaker embedding for each window from pretrained WavLM; drop the least typical 20% of windows if there are at least five, then average and L2-normalize the rest.
5. Compare the reference embedding and candidate embeddings using **cosine similarity** (nominal range **−1 to +1**). Sort descending, assign ranks, and export results. Any unreadable clips are reported at the end, not silently omitted.
6. Cache embeddings for standalone audio files using their resolved paths, sizes, modification times, and settings. Audio inside a ZIP is processed again on each run. Delete `.voice_ranker_cache/` to reset the cache.

### Interpreting a score

**0.83 means cosine similarity 0.83, NOT an 83% probability of matching and NOT 83% perceptual realism.** Scores can drift with microphone quality, background music, emotion, speaking style, and clip duration. There is no universally reliable threshold; calibrate on your own voice references, known good/known bad generated clips, and a few independent voices. Use results as a **ranking/filter**, not a final human verdict.

**Important for your reference clip:** A 30-second voice clip with some music versus a harshly denoised and choppy version is a real tradeoff. This tool deliberately avoids aggressive denoising; the VAD does not rewrite the voice's spectral content. Try both clips as references (two runs), and prefer candidates that rank well in both. If their ranking differs wildly, do not trust a single score.

### This is only one axis of a TTS judge

Speaker *identity* similarity is not the same as **human-perceived naturalness**, **emotion/prosody fidelity**, **speech intelligibility**, **distortion**, or **character/persona acting**. These require separate judges or human sampling. Do **not** automatically accept training samples solely because their speaker-similarity score is high.

## Practical troubleshooting

- **Can't load model / Hugging Face timeout:** Internet is needed for the one-time WavLM download. Retry when online.
- **Out of GPU memory:** Pass `--device cpu` (CLI) or choose `cpu` in the UI. This model should fit a typical RTX 4050 laptop for short clips, but exact VRAM use varies.
- **Only CPU despite NVIDIA GPU:** The installed PyTorch build may be CPU-only. Install a CUDA wheel compatible with your drivers using the official [PyTorch selector](https://pytorch.org/get-started/locally/) *inside `.venv`*. The tool will work on CPU meanwhile.
- **VAD says no voice:** Try turning it off; check whether the clip has actual speech.
- **M4A won't load:** Install FFmpeg or convert the clip to WAV.
- **Audio very noisy:** A high score doesn't certify high recording quality. Do not train on obviously broken audio just because its identity matches.
- **Sensitive audio:** All audio stays local; model downloads connect to Hugging Face. Do not set `share=True` in Gradio if privacy matters.

## Development / testing

```bash
python -m pip install pytest
python -m pytest -q tests
```

The included tests do not need a model download and validate audio decoding, sampling, CSV export, score ordering, cache behavior, and Gradio UI construction. **Actual speaker-verification quality must still be tested on real labeled audio after WavLM downloads.**

## References

- [Microsoft WavLM-Base-Plus for Speaker Verification](https://huggingface.co/microsoft/wavlm-base-plus-sv)
- [Hugging Face WavLM X-Vector API](https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/wavlm)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Gradio Files](https://www.gradio.app/main/docs/gradio/file)
