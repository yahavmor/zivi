# METAL VERSION — shared spec for all module builders

Project: a heavy modern-metal remix of a 5.5 s voice recording of the Hebrew sentence
"אתה מושעה מהמשחק, אתה מושעה, אתה מושעה מהמשחק!" ("you are suspended from the game").
The previous R&B remix lives in ../ (engine.py, vocal.py, midi_render.py, song.py). We are building a NEW
production (do not modify ../engine.py, ../vocal.py, ../midi_render.py, ../song.py). New modules live in THIS folder.

## Fixed musical decisions
- Sample rate 48000 Hz. All audio = numpy float64 arrays, mono shape (n,) or stereo shape (n,2). Peak of a
  single instrument render should be about 0.3–0.6 (headroom; mixing happens later). No NaNs, no DC offset.
- Tempo 110 BPM. beat = 0.54545 s, bar (4/4) = 2.18182 s. 16th = 0.13636 s.
- Key: C minor. Guitars in Drop C: low string C2 (MIDI 36, 65.4 Hz). Power chords = root + 5th + octave.
  Riff roots used: C2(36), Eb2(39), F2(41), G2(43), Ab2(44), Bb2(46), and octave-up versions.
- The voice sits around Bb3 (233 Hz); the big shouted syllable is ~C4; the sentence ends on G3.
- Structure (bars, 1-indexed, ~52 bars ≈ 113 s): 1–4 intro (dark, cinematic) · 5–8 first entry (sentence
  over sparse half-time drums + palm-muted chugs) · 9–16 main riff A (full band) · 17–24 riff B + lead melody
  + strings/choir · 25–28 build/tension (thins out, stutters, stop) · 29–36 BREAKDOWN (slow, huge, spaced
  chugs, demon-voice) · 37–40 lead solo over riff A · 41–48 final climax (everything) · 49–52 outro ring-out.

## Shared code you may use (import with `import sys; sys.path.insert(0, '..')`)
- `engine.py` (E): SR, midi2hz, db, stereo, mono, pan, gain, fade, adsr, env_exp, t_axis, read, write,
  biquad/eq(x, ('lp'|'hp'|'bp'|'notch'|'peak'|'ls'|'hs', f0, Q, gain_db), ...), lp, hp, lp_sweep_blocks,
  compress(x, thresh_db, ratio, attack, release, knee, makeup_db, sidechain), limiter(x, ceiling_db),
  saturate(x, drive, mix), softclip, reverb(x, decay, predelay, hf_damp, width, mix, lowcut, highcut, seed),
  delay(x, time, feedback, mix, pingpong, lowpass, highpass), chorus, tremolo_pan, widen, duck(x, times, ...),
  tape_stop, resample_speed, reverse, bitcrush, telephone, vinyl_noise, noise(n, seed), sine, saw_bl,
  square_bl, tri, scratch, Track(name, length_s).put(x, t, gain_db, pan_), ffmpeg_filter(in, out, af), measure(path).
- `midi_render.py` (M): render(events, program, out_path, bpm=120, channel=0, drums=False, gain=0.7,
  total_len=None) → stereo array rendered with FluidSynth + FluidR3_GM.sf2 (events = (t_sec, midi, dur_sec, vel0-1)).
  For drums pass drums=True (channel 9). To pick a drum KIT (e.g. Power kit = program 16) you must add a
  program_change on channel 9 — midi_render.render does not do that for drums; write your own small variant
  if you need it (mido is installed). FluidR3 kits: 0 Standard, 8 Room, 16 Power, 24 Electronic, 32 Jazz,
  48 Orchestra. GM melodic programs of interest: 29 Overdriven Gt, 30 Distortion Gt, 31 Gt harmonics,
  33 Finger bass, 34 Pick bass, 43 Contrabass, 44 Tremolo strings, 45 Pizzicato, 47 Timpani, 48/49 Strings,
  52 Choir Aahs, 53 Voice Oohs, 55 Orchestra Hit, 56 Trumpet, 57 Trombone, 58 Tuba, 60 French Horn,
  61 Brass, 14 Tubular Bells, 116 Taiko, 119 Reverse Cymbal, 47 Timpani.
- `vocal.py` (V): WORDS/PHRASES dicts with time boundaries inside the cleaned recording, slice_(x, a, b),
  rubberband(in, out, args) (Rubber Band CLI, R3 engine; `-p <semitones>`, `-F` formant preserve,
  `-t <ratio>` stretch), autotune(...), stutter(...). The cleaned mono voice is ../vox/clean.wav (5.53 s).
  Word slices already exist as ../vox/w_<name>.wav (ata1, mushe1, meham1, ata2, mushe2, ata3, mushe3, meham3,
  mu1, she1, me3, ha3, mis3, chak3, k_click, p1, p2, p3, p3a, full, p12).
- Tools on PATH: ffmpeg (filters incl. rubberband, afftdn, aexciter, acrusher, afir...), sox, rubberband, fluidsynth.
  No internet beyond PyPI. No real guitar/drum sample libraries exist: everything is synthesized in code or
  taken from the GM soundfont. Be honest in your report about what the result can and cannot sound like.

## Module contracts (each builder writes ONE file here + a smoke test + a short report)
All render functions take `events` = list of tuples with absolute time in seconds and return an array whose
length covers the last event + its tail (caller places it at t=0 on a Track). Keep functions pure
(no global state besides caches). Cache expensive one-shots to ./cache/*.wav. Keep rendering of a ~113 s song
under ~2 minutes per module on 1 CPU.

### metal_guitar.py
- `render_rhythm(events, seed=0, total_len=None) -> mono`  events: (t, root_midi, dur, vel, mute)
  mute=True → palm-muted chug (short, percussive, thumpy), False → ringing power chord.
  Synthesize a DI signal (e.g. Karplus-Strong / waveguide per string with pick transient, slight detune
  between the strings of the chord, realistic decay), THEN run the whole DI stream through an amp simulation
  (tight high-pass before distortion, multi-stage waveshaping with asymmetric clipping, tone stack with mid
  scoop/presence, noise gate) and a cabinet (synthesized 4x12 impulse response or an equivalent filter bank:
  strong 80–200 Hz, dip ~400–600 Hz, presence 2–4 kHz, steep roll-off above ~5–6 kHz).
- `render_rhythm_pair(events, total_len=None) -> stereo` : two independent takes (different seeds, ±3–6 ms
  timing jitter, slight pitch/pick differences) panned hard L/R = the "wall of sound".
- `render_lead(events, total_len=None) -> mono` events: (t, midi, dur, vel, bend_semis, vibrato, slide_from)
  single-note lead tone with long sustain, pitch bend (to bend_semis over the note), vibrato (depth in
  semitones, ~5.5 Hz), slide from another midi note. Lead amp = more gain and mids, less low end.
- `render_clean(events, total_len=None) -> stereo` events: (t, midi, dur, vel) clean/ambient guitar
  (arpeggios) with chorus + reverb, for intros/breakdown textures.
- extras welcome: `feedback_swell(midi, dur)`, `pick_scrape(dur)`, `pinch_harmonic(midi)`, `dive_bomb(midi, dur)`.

### metal_drums.py
- `hit(name, vel=1.0, seed=0) -> mono or stereo` one-shots for: kick, snare, snare_ghost, tom_hi, tom_mid,
  tom_floor, hat_closed, hat_open, hat_pedal, ride, ride_bell, crash1, crash2, china, splash.
  Layer synthesized elements (kick: sub body + beater click; snare: body + wires noise + crack layer) with
  FluidR3 GM Power-kit samples if they help (cymbals/toms from the soundfont are usually better than noise).
- `render_kit(events, total_len=None) -> stereo` events: (t, name, vel). Place hits with sensible panning
  (hats slightly L, ride R, toms spread, crashes L/R, kick/snare centre), velocity layers, round-robin
  variation, bus processing (kick/snare punch, parallel compression, overhead/room reverb). Must handle
  double-bass runs at 16ths (110 BPM) and 32nd fills without smearing.

### metal_bass.py
- `render_bass(events, total_len=None) -> mono`  events: (t, midi, dur, vel, slide_from, mute)
  Modern metal bass: clean sub band (<150 Hz: fundamental, tight) + distorted mid band (grit, pick attack,
  string noise) summed; palm-mute variant; slides (glide from slide_from over ~60 ms); compressed and tight.

### metal_orch_fx.py
- `orch(events, program, total_len=None, reverb_mix=0.35) -> stereo` FluidSynth wrapper + hall reverb.
- One-shots (mono or stereo): `drone(root_midi, dur)` (dark detuned low drone with slow movement),
  `riser(dur, f0, f1)`, `impact(vel)` (cinematic hit: sub boom + noise burst + metallic ring),
  `sub_drop(dur, f0=80, f1=25)`, `downlifter(dur)`, `metal_hit(vel, pitch_hz)` (inharmonic anvil/pipe),
  `reverse_cymbal(dur)`, `noise_bed(dur)` (dark atmospheric texture for intros), `braam(root_midi, dur)`
  (brass-like cinematic blast), `industrial_clank(seed)`.

### metal_vocal.py
- `variant(name, kind, **kw) -> stereo` for word/phrase names above and kinds:
  'clean' (EQ/compress for metal, no reverb), 'hall' (huge arena hall, 3–4 s), 'double' (two detuned/timed
  copies L/R), 'demon' (−12 semitones formant-preserved + distortion + lowpass), 'shout_stack' (centre +
  quiet octave-down + quiet octave-up + saturation = a metal frontman stack), 'megaphone', 'radio',
  'stutter' (16th/32nd repeats at 110 BPM), 'reverse_swell' (reverse-reverb into the word),
  'glitch' (random micro-slices), 'slowmo' (half speed), 'tele'.
  Reuse ../vox/w_*.wav and vocal.py; cache outputs in ./cache.

## Reporting
Finish by returning: the public function signatures you implemented, how you validated them (numbers:
peak, RMS, spectral balance; spectrogram image path if you made one), and honest limitations.
