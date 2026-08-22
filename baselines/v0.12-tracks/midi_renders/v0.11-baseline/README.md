# LSD-studio reference renders (v0.11-era)

Copied 2026-08-22 from the studio app data directory
(`~/Library/Application Support/lsd-studio/output-generation/<session>/`),
newest renders at copy time. Filenames are the app's own provenance:
worker tag (`v0.3.1` / `lsd-0.11.0`), MIDI pattern tag
(`melodic_4b` = 4-bar melodic, `bass_8b` = 8-bar bass), checkpoint hash.

Purpose: A/B reference for `../materialised_*.wav` (v0.12 native stems
materialised by scripts/materialise_midi.py). Note these were produced
from models trained on the producer archive in-app, not on ESC-50 —
timbre differences are expected; listen for render-quality dimensions
(pitch tracking, clicks, note tails).

Provenance rule (per owner): **the shortest-named file(s) are the ones
produced directly by the model** —
`render_*_midi_*_melodic_4b__410e2fb0_v0.3.1_*.wav`. Longer names that
embed another render's filename (e.g.
`..._melodic_4b__render_1786212757595_....wav`) are derivatives /
re-renders of earlier outputs, not fresh model draws.
