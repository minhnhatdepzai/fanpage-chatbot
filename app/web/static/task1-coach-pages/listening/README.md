# Listening audio — provenance

These 30 MP3 files are synthetic narration of the original practice scripts in
`src/data/listening.ts`, `listeningExpansion.ts` and `listeningCloze.ts`. They are not official IELTS/Cambridge recordings, human
interviews, or a complete timed mock exam. Both dialogue roles use the same voice.
No user-provided transcript is sent to a speech service.

Generated locally with Piper 1.8.0 and `en_GB-alba-medium`, length scale 1.12,
0.35-second sentence silence; converted with ffmpeg to mono MP3, 80 kbps / 24 kHz,
loudness normalized to -18 LUFS. Use `scripts/generate-listening-audio.ts` to rebuild.
Piper and its model are development tools, not distributed in this application.

Voice model card:
https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_GB/alba/medium/MODEL_CARD

Model-card dataset attribution (CC BY 4.0):
Cassia Valentini-Botinhao and Junichi Yamagishi, University of Edinburgh (2019),
“English multi-speaker corpus for CSTR voice cloning toolkit (version 0.92)”.
https://doi.org/10.7488/ds/2506
https://datashare.ed.ac.uk/handle/10283/3270
https://creativecommons.org/licenses/by/4.0/

This acknowledgement does not imply endorsement by the dataset authors or speaker.
The recordings here were synthesized from new scripts, not copied from the corpus.
The model card also notes fine-tuning from en_US-lessac-medium. No training data or
model weights are included in this repository.

Engine source (GPL-3.0): https://github.com/OHF-Voice/piper1-gpl

Durations for the first four lessons (including the brief introductory instruction):

- riverside-booking.mp3 — 108.432 seconds
- community-garden.mp3 — 122.256 seconds
- student-research.mp3 — 120.552 seconds
- urban-trees.mp3 — 143.160 seconds
