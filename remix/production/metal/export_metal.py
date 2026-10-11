"""Export the metal mixes as WAV 44.1k/16 + MP3 320 with cover, into the repo (remix/metal)."""
import subprocess, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('METAL_OUT', os.path.join(HERE, '..', '..', 'out', 'metal'))
DEST = sys.argv[1] if len(sys.argv) > 1 else "/home/user/zivi/remix/metal"; os.makedirs(DEST, exist_ok=True)
TITLES = {'mushe_metal': ('מושעה מהמשחק (Metal Version)', 'Mushe_Metal_Version'),
          'mushe_metal_instrumental': ('מושעה מהמשחק (Metal Version, Instrumental)', 'Mushe_Metal_Version_Instrumental')}
cover = os.path.join(OUT, 'cover_metal.jpg')
for key, (title, fname) in TITLES.items():
    src = os.path.join(OUT, key + '_48k.wav')
    if not os.path.exists(src): print("missing", src); continue
    wav = os.path.join(DEST, fname + '.wav'); mp3 = os.path.join(DEST, fname + '.mp3')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', src, '-ar', '44100', '-af', 'aresample=dither_method=triangular', '-sample_fmt', 's16', wav], check=True)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', src, '-i', cover, '-map', '0:a', '-map', '1:v', '-c:a', 'libmp3lame', '-b:a', '320k', '-ar', '44100', '-c:v', 'mjpeg', '-id3v2_version', '3',
                    '-metadata', f'title={title}', '-metadata', 'artist=Yahav (prod.)', '-metadata', 'album=מושעה מהמשחק', '-metadata', 'genre=Metal', '-metadata:s:v', 'title=Album cover', '-metadata:s:v', 'comment=Cover (front)', mp3], check=True)
    print("exported", wav, mp3)
shutil.copy(cover, os.path.join(DEST, 'cover_metal.jpg'))
