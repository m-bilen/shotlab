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
- [x] Pretrained YOLO comparison: yolo26m at 640 finds the ball in 91% of frames around release
- [x] Release detection: 67/67 within 2 frames of manual labels, rebounds and gestures rejected

## Next (January release)

- [ ] Dec 1-5: YOLO fine-tune: small model with yolo26m accuracy, at least 3x faster
- [ ] Dec 6-12: Labeling tool (made / missed / not a shot / unclear audio)
- [ ] Dec 6-12: Manual make/miss labels for S01-S04
- [ ] Dec 13-19: Audio label model, auto-label S05-S08, blind review
- [ ] Dec 20-26: Features (smoothing, release interpolation, calibration, measurement error) and shot table
- [ ] Dec 27-Jan 5: Models and evaluation, A1-A4 (see analysis_plan.md)
- [ ] Jan 6-12: Findings, plots, README, demo GIF

## Later
- [ ] Experiment (A-B-B-A)
- [ ] Automatic make/miss from ball and rim
- [ ] FastAPI + SQLite backend
- [ ] Automatic shooter region with YOLO person detection
- [ ] Docker image and docker compose for the backend
- [ ] React results page on GitHub Pages
- [ ] Wrap pose landmarker and its running mode in a class (when a third use appears)