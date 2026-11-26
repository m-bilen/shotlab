# Design decisions

- **Free throws instead of jump shots:** knee injury; also a more repeatable
  motion.
- **Fixed square region around the shooter:** MediaPipe could not detect a
  small person in the full 1920x1080 frame; cropping fixed this and excluded
  the rebounder.
- **Sequential frame reading:** phone videos have variable frame rate (dropped
  frames in every session, most in S03, S04 and S07; light alone does not
  explain it), so seeking by frame number landed on wrong frames.
- **Heavy pose model:** offline processing, so accuracy matters more than
  speed (~20 fps on CPU).
- **Manual make/miss labels instead of Whisper:** Whisper missed isolated
  one-word labels ("muz"/"elma") spoken 6-9 m from the phone, even with
  hotwords, VAD and an initial prompt.
- **Shot detection rule:** a shot is a period when the right wrist is above the
  nose (gaps < 120 ms merged, periods < 250 ms dropped) and the elbow opens to
  at least 150 degrees. Catching a pass usually lifts the wrist but keeps the
  elbow bent. Elbow angle at the wrist's highest point did not work: the
  father's wrist peaks at his set point, before the arm extends.
- **Manual labels are cross-checked against detection:** 4 of 58 manual
  release frames did not fall inside any detected shot. All 4 were labeling
  mistakes (no shot at that frame) and were corrected. Manual labels are not
  assumed to be error-free.
- **Pose jump threshold in body units:** a jump is a frame-to-frame speed above
  30 torso lengths per second (right shoulder to right hip, median per video).
  Torso length ranged from 99 to 151 px across videos, so a pixel threshold
  meant a different limit in each video. The 99th percentile of wrist speed was
  7-9 torso lengths/s in every video, at both 30 and 60 fps. 30 is about twice
  the fastest real wrist motion in a free throw.
- **Known false shots:** catching a rebound overhead with a straight arm, or
  pointing with the arm up, also opens the elbow past 150 degrees. Shots less
  than 3 s apart were reviewed: all were a rebound or a gesture next to a real
  shot (about 1-2% of detections). Not fixed in the pose rule: a catch and a
  shot look alike in pose but differ in ball direction. Release detection with
  the ball and manual review ("not a shot") remove them.
- **Long detections are real shots:** 3-4 s detections were a pause at the set
  point or a held follow-through. Release is searched in the whole window, not
  at its end.
- **Release frame (manual):** the first frame where a gap between the ball and
  the fingers is clearly visible. If the ball still seems to touch the fingers,
  the next frame is taken. So manual frames may be up to 1 frame later than the
  true release.
- **Manual release frames re-checked blind:** 20 shots (all 6 of S02 and 2
  random shots from every other session) were labeled again without seeing the
  old labels or the automatic result. Labels made after the rule was written
  matched within 1 frame (14 of 15 identical). The first 4 labels, made on the
  first day, were 3-4 frames early. All 20 re-labeled shots use the new label.
- **Gross label errors found by release detection:** with the original labels,
  64 of 67 automatic releases were within 2 frames. The 3 misses (all S03) were
  labeling mistakes: the manual frame showed the follow-through, after the ball
  was gone. They were re-labeled by the written rule, but not blind (the
  automatic frame had been seen). S03 had 7 wrong labels in total out of 15.
- **Release from ball-to-fingertip distance:** release is the first moment the
  ball center is more than 1.2 ball sizes (median per shot) from the right
  index finger and stays there for 100 ms, interpolated between frames. Ball
  from pretrained yolo26m at 640 in the shooter region extended up. Search
  window: detection start -200 ms to detection end +200 ms (a window around the
  wrist peak missed 3 of 67 releases). The leave-one-session-out threshold was
  1.15-1.2 in every fold. Rebounds 0/5 and gestures 0/2 got a release, real
  shots 9/9.
- **No audio label model:** every shot is labeled by hand, because make/miss
  is the target and a model's confident mistakes would enter the analysis
  without anyone seeing them. With that rule an audio model saves no labeling,
  so it was dropped. Blind re-labels measure label errors instead.
- **Fine-tune goal is CPU speed:** the full dataset with yolo26m takes about
  11 hours on this laptop's CPU. A fine-tuned small model should do it in about
  2 hours. Training runs on Colab.
- **Training frames for ball fine-tuning:** taken from shot start -300 ms to
  +500 ms, not the whole detection. In the whole detection, more than half of
  the frames came after the ball had left the region (teacher found a ball in
  46% of them). Release is about 80 ms after shot start for Mehmet but about
  480 ms for his father, who holds the ball at the set point. So the training
  window caught few frames after the father's release. The student still
  passed all 20 of his exam shots.
- **Fine-tuned yolo26n for the ball:** trained on 1568 reviewed crops from
  S01-S07, 100 epochs on a Colab T4 with default settings; the best epoch was
  picked on 320 crops from S08. On the release exam it matched yolo26m: 67/67
  within 2 frames (95% interval 94.6-100%), 64/64 on blind labels only, a
  release for 0 of 7 rebounds and gestures, 7/7 on S08 which it never trained
  on. Release threshold 1.15 (1.15-1.2 leave-one-session-out). Same CPU, same
  190 crops: 52 ms vs 228 ms per image (4.4x); with video reading included,
  about 3-3.5x.
- **Teacher quality:** on frames with a ball, the teacher's box was kept for
  83%, redrawn for 9% and missing for 8%. It almost never boxed a ball that was
  not there (3 frames).
- **Ball class from the model:** the code first asked for COCO class 32
  ("sports ball"). The fine-tuned model has a single class 0, so it found
  nothing. The ball class is now read from the model's own class names.
- **Release for every shot:** the fine-tuned yolo26n on every detection window
  (detection start -200 ms to end +200 ms), threshold 1.15: 2,033 of 2,073
  detections got a release, in about 2.5 hours on CPU.
- **Full-run audit:** 9 releases came less than 3 s after another release, in
  6 sequences. In all 6, a rebound had got a release, so rebound rejection is
  not perfect (0/5 on the exam was a small sample). Every rebound that gets a
  release sits within 3 s of a real shot, so this check finds them, and
  make/miss labeling marks them "not a shot". All 12 detections without a
  release that were not near another detection were warm-up moves or rebounds:
  no real shot lost there. These shots are kept apart from the exam set
  (audit_shots.csv), because they were picked by looking at the result.
- **Full-run release check (blind):** 40 random released shots (20 per
  shooter) outside the exam and the audit, labeled blind. 4 of them turned out
  to be in the ball training set (the sample did not exclude it). On the other
  36: 36/36 within 2 frames (95% interval 90-100%), 18/18 per shooter, all
  within 1.4 frames. The 4 training shots were also within 2 frames. With the
  exam, 103/103 unseen shots. Two labels were typos (an extra or a missing
  digit); they were re-labeled from the video, not copied from the automatic
  result. From now on, check samples exclude training shots in code.
- **Renamed to Swish or Brick:** a commercial coaching service already uses
  "ShotLab". Renamed before links and the README grew.
- **Analysis plan rewritten before labeling:** after two rounds of outside
  review and before any make/miss label existed. Features listed and frozen.
  Knee bend measured from before shot start, since shot start is a detector
  event and the dip comes earlier; feature 7 became knee-to-release time.
  Landmarks moved to an even time grid before smoothing, because frames drop.
  Wrist speed over 100 ms, smoothing fixed in time units so 30 and 60 fps
  measure the same thing. A4 uses absolute deviation with median and MAD within
  each shooter and session, a block-shuffle test, shot number and
  session-shooter terms. "Best range" got an exact procedure (5 bins chosen on
  training sessions, best bin vs the rest in the held-out session, one-sided,
  block shuffle, Holm). A1a uses within-session AUC and a shot-number baseline,
  because pooled AUC can rise from session differences alone. A3 compares
  shooters within sessions and is descriptive. Label checks: 10% with sound
  and 100 shots muted, since a relabel with sound cannot catch a wrong word
  said at recording time. "Can't tell" is checked as all makes and all misses.
  The A-B-B-A experiment is conditional, sized for ~9 points, with a
  manipulation check and block-level test.
- **Knee bend window:** checked before the freeze on 16 random shots per
  shooter, without labels. The deepest knee bend came FILL s (median) before
  shot start for Mehmet and FILL s for his father; FILL of 32 had a pre-shot
  move (bounce, bend) inside the window. W = FILL s.