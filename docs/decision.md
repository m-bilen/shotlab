# Design decisions

- **Free throws instead of jump shots:** knee injury; also a more repeatable motion.
- **Fixed square region around the shooter:** MediaPipe could not detect a small person in the full 1920x1080 frame; cropping fixed this and excluded the rebounder.
- **Sequential frame reading:** phone videos have variable frame rate in low light, so seeking by frame number landed on wrong frames.
- **Heavy pose model:** offline processing, so accuracy matters more than speed (~20 fps on CPU).
- **Manual make/miss labels instead of Whisper:** Whisper missed isolated one-word labels ("muz"/"elma") spoken 6-9 m from the phone, even with hotwords, VAD and an initial prompt.