# Roadmap

## Done
- [x] Data collection: 8 sessions, ~1000 shots each (Mehmet, father)
- [x] Setup: Python, venv, Git, GitHub, CI
- [x] Video reading, frame-by-frame player, landmark viewer
- [x] Pose extraction (MediaPipe heavy) with fixed shooter region
- [x] Landmark quality report
- [x] Session log with fps and shooter boundaries
- [x] Shot detection: wrist above nose, catches removed by elbow extension
- [x] Manual release frames: 67 shots, every session and both shooters
- [x] Quality report independent of camera distance and fps
- [x] Shot detection reviewed: false shots (rebounds, gestures) and long shots documented

## Next (January release)
- [ ] Nov 22-24: Pretrained YOLO, ball detection rate per session
- [ ] Nov 25-29: Release time from ball trajectory, compared with manual frames
- [ ] Nov 25-29: Check that release detection rejects the reviewed rebound detections
- [ ] (+1 week if needed) YOLO fine-tune
- [ ] Nov 30-Dec 6: Labeling tool (made / missed / not a shot / unclear audio)
- [ ] Nov 30-Dec 6: Manual make/miss labels for S01-S04
- [ ] Dec 7-13: Audio label model, auto-label S05-S08, blind review
- [ ] Dec 14-20: Features (smoothing, release interpolation, calibration, measurement error) and shot table
- [ ] Dec 21-Jan 3: Models and evaluation, A1-A4 (see analysis_plan.md)
- [ ] Jan 4-10: Findings, plots, README, demo GIF

## Later
- [ ] Experiment (A-B-B-A)
- [ ] Automatic make/miss from ball and rim
- [ ] FastAPI + SQLite backend
- [ ] Automatic shooter region with YOLO person detection
- [ ] Docker image and docker compose for the backend
- [ ] React results page on GitHub Pages
- [ ] Wrap pose landmarker and its running mode in a class (when a third use appears)