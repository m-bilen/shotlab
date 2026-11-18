# Design decisions

- **Free throws instead of jump shots:** knee injury; also a more repeatable motion.
- **Fixed square region around the shooter:** MediaPipe could not detect a small person in the full 1920x1080 frame; cropping fixed this and excluded the rebounder.
- **Sequential frame reading:** phone videos have variable frame rate in low light, so seeking by frame number landed on wrong frames.
- **Heavy pose model:** offline processing, so accuracy matters more than speed (~20 fps on CPU).
- **Manual make/miss labels instead of Whisper:** Whisper missed isolated one-word labels ("muz"/"elma") spoken 6-9 m from the phone, even with hotwords, VAD and an initial prompt.

- **Shot detection rule:** a shot is a period when the right wrist is above the nose
  (gaps < 120 ms merged, periods < 250 ms dropped) and the elbow opens to at least
  150 degrees. Catching a high pass also lifts the wrist but keeps the elbow bent.
  Elbow angle at the wrist's highest point did not work: the father's wrist peaks
  at his set point, before the arm extends.