# Roadmap

## Done
- [x] Data collection: 8 sessions, ~1000 shots each (Mehmet, father)
- [x] Setup: Python, venv, Git, GitHub, CI
- [x] Video reading, frame-by-frame player, landmark viewer
- [x] Pose extraction (MediaPipe heavy) with fixed shooter region
- [x] Landmark quality report
- [x] Session log with fps and shooter boundaries
- [x] Shot detection: wrist above nose, catches removed by elbow extension
- [x] Manual release frames: 58 shots across all sessions

## Next (January release)
- [ ] Pretrained YOLO: ball detection rate per session
- [ ] Release time from ball trajectory, compared with manual frames
- [ ] YOLO fine-tune, if pretrained is not good enough
- [ ] Labeling tool (made / missed / not a shot / unclear audio)
- [ ] Manual make/miss labels for S01-S04
- [ ] Audio label model, auto-label S05-S08, blind review
- [ ] Features: smoothing, release interpolation, calibration, measurement error
- [ ] Shot table
- [ ] Models and evaluation: A1-A4 (see analysis_plan.md)
- [ ] Findings, plots, README, demo GIF

## Later
- [ ] Experiment (A-B-B-A)
- [ ] Automatic make/miss from ball and rim
- [ ] FastAPI + SQLite backend
- [ ] Automatic shooter region with YOLO person detection
- [ ] Docker image and docker compose for the backend
- [ ] React results page on GitHub Pages