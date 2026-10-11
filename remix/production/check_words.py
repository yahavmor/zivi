"""Run whisper on windows of the final mix to verify the words survive the mix."""
import sherpa_onnx, soundfile as sf, numpy as np, sys, os
from scipy.signal import resample_poly
M = os.environ.get("WHISPER_MODELS", "models")  # sherpa-onnx whisper-turbo folder parent
rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=f"{M}/sherpa-onnx-whisper-turbo/turbo-encoder.int8.onnx", decoder=f"{M}/sherpa-onnx-whisper-turbo/turbo-decoder.int8.onnx", tokens=f"{M}/sherpa-onnx-whisper-turbo/turbo-tokens.txt", language="he", task="transcribe", num_threads=4)
x, sr = sf.read(sys.argv[1]); m = x.mean(1) if x.ndim == 2 else x
m16 = resample_poly(m, 1, 3).astype('float32')  # 48k -> 16k
BAR = 240 / 72
wins = [(1, 3), (3, 5), (5, 9), (9, 11), (11, 13), (13, 15), (15, 17), (17, 19), (19, 21), (21, 23), (23, 25), (25, 27)]
for a, b in wins:
    s = int((a - 1) * BAR * 16000); e = int(min(len(m16), (b - 1) * BAR * 16000))
    seg = np.concatenate([np.zeros(6400, 'float32'), m16[s:e], np.zeros(6400, 'float32')])
    st = rec.create_stream(); st.accept_waveform(16000, seg); rec.decode_stream(st)
    print(f"bars {a:2d}-{b-1:2d}: {st.result.text!r}")
