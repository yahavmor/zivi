"""Render note events through FluidSynth (FluidR3 GM) to a stereo wav."""
import mido, subprocess, os, numpy as np
import engine as E
SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
def render(events, program, out_path, bpm=120, channel=0, drums=False, gain=0.7, total_len=None):
    """events: list of (t_sec, midi_note, dur_sec, vel 0-1). returns stereo array (len fixed to total_len if given)."""
    mid = mido.MidiFile(ticks_per_beat=960); tr = mido.MidiTrack(); mid.tracks.append(tr)
    tempo = mido.bpm2tempo(bpm); tr.append(mido.MetaMessage('set_tempo', tempo=tempo))
    ch = 9 if drums else channel
    if not drums: tr.append(mido.Message('program_change', program=program, channel=ch, time=0))
    tr.append(mido.Message('control_change', control=91, value=0, channel=ch, time=0))  # no reverb
    tr.append(mido.Message('control_change', control=93, value=0, channel=ch, time=0))  # no chorus
    msgs = []
    for (t, n, d, v) in events:
        msgs.append((t, 1, mido.Message('note_on', note=int(n), velocity=int(np.clip(v * 127, 1, 127)), channel=ch)))
        msgs.append((t + d, 0, mido.Message('note_off', note=int(n), velocity=0, channel=ch)))
    msgs.sort(key=lambda m: (m[0], m[1]))
    last = 0.0
    for t, _, m in msgs:
        ticks = int(round(mido.second2tick(t - last, 960, tempo))); m.time = max(0, ticks); tr.append(m); last = t
    end = (total_len or (last + 2.0)); tr.append(mido.MetaMessage('end_of_track', time=int(mido.second2tick(max(0, end - last), 960, tempo))))
    midp = out_path.replace('.wav', '.mid'); mid.save(midp)
    subprocess.run(['fluidsynth', '-ni', '-q', '-R', '0', '-C', '0', '-g', str(gain), '-r', str(E.SR), '-F', out_path, SF2, midp], check=True, capture_output=True)
    x = E.read(out_path)
    if total_len is not None:
        n = int(total_len * E.SR); y = np.zeros((n, 2)); m = min(n, len(x)); y[:m] = x[:m]; x = y
    return x
