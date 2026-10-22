# face-tweaker

This project fits The Division 2 face sliders to a target photo. It works only from recorded video of the character creator.

## Game safety

The Division 2 ships Easy Anti-Cheat and is always online. A ban on the user's account outweighs any data gain, so the project never touches the running game:

- No D3D capture tools (RenderDoc, Ninja Ripper).
- No OBS Game Capture, because it injects a hook.
- No scripted key or mouse input.

Propose only offline analysis of recordings. The user records with the NVIDIA app (Alt+F9) and moves the sliders by hand.

## Recording rules

- Clips go in `C:/Users/LexSong/Videos/NVIDIA/Tom Clancy's The Division 2/HeadNN/`. The game window is 16:9 and clips are 1280x720, 60 fps. 4K files were too large, and 720p reads the same.
- One fixed slider setting per clip. The sliders must not move during a clip.
- Do not steer the head with the mouse. A steered clip gave worse idle animation axes than a plain idle clip.
- The idle animation loops every ~17.5 s, so a clip needs at least one full loop.
- The slider panel must not scroll, or the clip is rejected. So Jawline is not read.
- Heads 07, 12, 14 and 15 are skipped.

## Workflow per head

1. Control clip: all 9 sliders at 6, ~35 s.
2. Sweep: each slider at 0 and at 12, the rest at 6, ~20 s each. That is 18 clips.
3. Run, in order:
   - `uv run scripts/extract_features.py <recordings> data/features` (it checks the new clips first)
   - `uv run scripts/fit_sliders.py data/features <head> data/images/*`
4. Record the suggestion as one clip in the same head folder, then refit. Repeat until the suggestion stops changing.

Recording time is the user's main cost, so sweep only the 3-4 heads that rank highest for the target.

## Measurement rules

- Measure faces with ArcFace (buffalo_l w600k_r50). MediaPipe 3D points and 3DDFA_V2 followed the sliders 3-5x worse, so do not propose them again. MediaPipe stays only for head yaw and pitch.
- Keep frames with |yaw| <= 25 degrees. Far from the resting pose, ArcFace changes with the angle.
- Remove each head's idle animation with its own idle animation axes. Learn them on clips that are not scored, or the noise estimate is circular.
- Sessions drift: the same head moved 0.05-0.09 between sessions, 2-7x the in-clip noise. So never fit a clip against a control from another session.
- Fit one local model per head and refine it with new clips. The user rejected one global least-squares fit.
- Judge any change to the frame or clip rules by how well the kept frames agree.
