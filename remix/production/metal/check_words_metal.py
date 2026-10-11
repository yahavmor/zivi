import sherpa_onnx, soundfile as sf, numpy as np, sys, json, os
from scipy.signal import resample_poly
M = os.environ.get("WHISPER_MODELS", "models")  # folder holding sherpa-onnx-whisper-turbo
rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=f"{M}/sherpa-onnx-whisper-turbo/turbo-encoder.int8.onnx", decoder=f"{M}/sherpa-onnx-whisper-turbo/turbo-decoder.int8.onnx", tokens=f"{M}/sherpa-onnx-whisper-turbo/turbo-tokens.txt", language="he", task="transcribe", num_threads=4)
x, sr = sf.read(sys.argv[1]); m = x.mean(1); m16 = resample_poly(m, 1, 3).astype('float32'); BAR = 240 / 110
riffs = json.load(open(os.environ.get('METAL_RIFFS', 'riffs.json'))); ORDER = ['intro','first_entry','riffA','riffB','build','breakdown','solo','climax','outro']
b = 1; wins = []
for s in ORDER: wins.append((s, b, b + riffs['sections'][s]['bars'])); b += riffs['sections'][s]['bars']
for s, a, e in wins:
    seg = np.concatenate([np.zeros(6400, 'float32'), m16[int((a - 1) * BAR * 16000):int(min(len(m16), (e - 1) * BAR * 16000))], np.zeros(6400, 'float32')])
    st = rec.create_stream(); st.accept_waveform(16000, seg); rec.decode_stream(st); print(f"{s:12s} bars {a:2d}-{e-1:2d}: {st.result.text!r}")
