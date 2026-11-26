# Live Class Assistant

Records a class, transcribes it live and tells apart the **teacher (T)** from the **students (S)**. Everything runs locally on the Mac.

Full design and decisions (in Spanish): [DISENO.md](DISENO.md).

**Status:** phase 1 done (live transcription + speakers + history + audio). Next: phase 2 (ask the AI about the class).

## Setup (once)

```bash
cd clase-ia
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p models
curl -L -o models/silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
curl -L -o models/speaker.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
```

On first run, the Parakeet v3 model (~1.2 GB) is downloaded to `~/.cache/huggingface`.

## Usage

```bash
.venv/bin/python main.py --subject "Calculus"
```

- Opens `http://127.0.0.1:8765/` (full page) and `/mini` (small window).
- **Ctrl+C** or the "End class" button closes the class. The text, the audio and the teacher's voiceprint are saved.
- On first run, macOS asks for **microphone permission** for the Terminal.

Other options:

```bash
.venv/bin/python main.py --devices                                  # list microphones
.venv/bin/python main.py --subject "Calculus" --device "USB"        # external microphone
.venv/bin/python main.py --subject Demo --file tests/class_sample.wav   # simulate a class
```

Settings live in [config.yaml](config.yaml): microphone, VAD pauses, speaker thresholds, port and data folder.

## How it works

1. **Audio:** microphone or file at 16 kHz mono, also saved as Opus (~11 MB per hour).
2. **VAD:** silero (sherpa-onnx) cuts the audio into sentences at each pause.
3. **Transcription:** Parakeet v3 (MLX) transcribes each full sentence, about 20× faster than real time. While someone is speaking, the sentence in progress is re-transcribed every 1.5 s as provisional text.
4. **Speakers:** a voice embedding (CAM++, sherpa-onnx) groups sentences by voice. That is combined with text cues ("abrid el cuaderno" vs. "profe, una pregunta") and turn-taking (a teacher's "good question" marks the previous sentence as a student's) into a probability P(teacher).
   - The teacher's voiceprint is saved per subject, so the next class recognises them from the first sentence.
5. **Storage:** SQLite with full-text search (FTS5), plus a local web page updated over a WebSocket.

## Data

| Path | Contents |
|---|---|
| `data/class.db` | SQLite: subjects, classes, sentences (with FTS5 search) and the teacher voiceprint per subject |
| `data/audio/*.opus` | Audio of each class |

## Tests

```bash
tests/make_test_audio.sh                    # builds a sample class with two system voices
.venv/bin/python tests/test_pipeline.py     # runs it through the pipeline, without the web page
```

## Measured performance (MacBook Air M1, 8 GB)

- **Transcription:** about 20× faster than real time.
- **Final sentence:** appears ~1–1.5 s after it is spoken (0.5 s pause + 0.3–0.9 s processing).
- **Provisional text:** refreshed every 1.5 s while someone is speaking.
- **RAM:** ~1.4 GB while running (2.5 GB peak while loading the model).
