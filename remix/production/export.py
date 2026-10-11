"""Export WAV 44.1k/16 + MP3 320 with cover art, into the repo folder."""
import subprocess, sys, os, shutil
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); os.makedirs(DEST, exist_ok=True)
TITLES = {'mushe_remix': ('מושעה מהמשחק (Late Night R&B Edit)', 'Mushe_Late_Night_RnB_Edit'),
          'mushe_remix_instrumental': ('מושעה מהמשחק (Instrumental)', 'Mushe_Late_Night_RnB_Edit_Instrumental'),
          'mushe_remix_extra_comedy': ('מושעה מהמשחק (Extra Comedy Edit)', 'Mushe_Late_Night_RnB_Edit_ExtraComedy')}
cover = f"{OUT}/cover.jpg"
for key, (title, fname) in TITLES.items():
    src = f"{OUT}/{key}_48k.wav"
    if not os.path.exists(src): print("missing", src); continue
    wav = f"{DEST}/{fname}.wav"; mp3 = f"{DEST}/{fname}.mp3"
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', src, '-ar', '44100', '-af', 'aresample=dither_method=triangular', '-sample_fmt', 's16', wav], check=True)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', src, '-i', cover, '-map', '0:a', '-map', '1:v', '-c:a', 'libmp3lame', '-b:a', '320k', '-ar', '44100', '-c:v', 'mjpeg', '-id3v2_version', '3',
                    '-metadata', f'title={title}', '-metadata', 'artist=Yahav (prod.)', '-metadata', 'album=מושעה מהמשחק', '-metadata', 'genre=R&B', '-metadata:s:v', 'title=Album cover', '-metadata:s:v', 'comment=Cover (front)', mp3], check=True)
    print("exported", wav, mp3)
shutil.copy(cover, f"{DEST}/cover.jpg")
