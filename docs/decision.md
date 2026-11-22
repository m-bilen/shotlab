# Design decisions

- **Free throws instead of jump shots:** knee injury; also a more repeatable motion.
- **Fixed square region around the shooter:** MediaPipe could not detect a small person in the full 1920x1080 frame; cropping fixed this and excluded the rebounder.
- **Sequential frame reading:** phone videos have variable frame rate (dropped
  frames in every session, most in S03, S04 and S07; light alone does not explain
  it), so seeking by frame number landed on wrong frames.
- **Heavy pose model:** offline processing, so accuracy matters more than speed (~20 fps on CPU).
- **Manual make/miss labels instead of Whisper:** Whisper missed isolated one-word labels ("muz"/"elma") spoken 6-9 m from the phone, even with hotwords, VAD and an initial prompt.
- **Shot detection rule:** a shot is a period when the right wrist is above the nose
  (gaps < 120 ms merged, periods < 250 ms dropped) and the elbow opens to at least
  150 degrees. Catching a pass usually lifts the wrist but keeps the elbow bent.
  Elbow angle at the wrist's highest point did not work: the father's wrist peaks
  at his set point, before the arm extends.
  - **Manual labels are cross-checked against detection:** 4 of 58 manual release
  frames did not fall inside any detected shot. All 4 were labeling mistakes
  (no shot at that frame) and were corrected. Manual labels are not assumed
  to be error-free.
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