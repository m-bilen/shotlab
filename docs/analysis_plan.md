# Analysis plan

Single source of truth for data, splits and leakage rules.
Any change to the analysis is written here first.

## 1. Data

| Session | Date | Camera | FPS | Light | Mehmet | Father |
|---|---|---|---|---|---|---|
| S01 | 2026-11-11 | 9 m | 60 | electric | ~150 | ~150 |
| S02 | 2026-11-11 | 6 m | 60 | electric | ~50 | ~50 |
| S03 | 2026-11-12 | 7.5 m | 60 | daylight | ~150 | ~150 |
| S04 | 2026-11-12 | 6.5 m | 60 | dark | ~100 | ~100 |
| S05 | 2026-11-14 | 7 m | 30 | electric | ~150 | ~195 (2 videos) |
| S06 | 2026-11-14 | 7 m | 30 | electric | ~150 | ~150 |
| S07 | 2026-11-15 | 7 m | 60 | electric | ~150 | ~150 |
| S08 | 2026-11-15 | 7 m | 60 | electric | ~100 | ~100 |

- Unit of analysis: one shot.
- Shot type: free throw, no jump, same spot.
- Label: made / missed. Ground truth = manual labels.
- Mehmet ~900 shots, father ~950 shots.

## 2. Analyses

**A1. Mehmet's best form** (main result)
- Data: Mehmet's shots only.
- Question: which release mechanics are linked to made shots?

**A2. Father's best form**
- Same as A1 on father's shots.

**A3. Comparison**
- Are the best ranges in A1 and A2 different? Is ideal form personal?

**A4. Consistency** (pooled, both shooters)
- Each feature expressed as deviation from that shooter's own mean.
- Question: does deviating from your own normal form lower the make rate?
- Shooter is a model input; raw (not normalized) features are not.

## 3. Evaluation

- Split: leave-one-session-out. All shots of the held-out session,
  from both shooters, are excluded from training.
- Models: logistic regression, random forest.
- Baselines: always predict the majority class; shooter's base make rate.
- Metrics: ROC AUC, log loss, accuracy.
- Report results separately for 30 fps and 60 fps sessions.
- Smallest reliably detectable effect with ~900 shots: about 8-10
  percentage points in make rate.
- Expected effect is small: a free throw without a jump is a consistent motion.
  A null result is reported as such, together with the smallest effect we could
  have detected. A4 (consistency) is the most likely place for a signal.

## 4. Leakage rules

1. **Split by session, never by random shots.** Shots in one session share
   light, camera and fatigue.
2. **Only movement up to the release moment is a feature.** Nothing after
   release: the shooter reacts to the outcome. `duration_s` from shot
   detection is NOT a feature.
3. **Every fitted step uses training sessions only:** scaling, feature
   selection, "best range" thresholds, per-shooter means in A4.
   Use scikit-learn Pipeline.
4. **No session-identifying inputs:** camera distance, fps, date, light.
   Used only to check results per group.
5. **Remove duplicate shots** before any analysis.
6. **Release detection:** one threshold (ball-to-fingertip distance in ball
   sizes). Chosen leave-one-session-out on manual release frames: each session
   is tested with a threshold chosen on the other sessions. Frames after
   release are used only to find the release moment, never as features. The
   search window around the wrist peak was set after looking at all manual
   frames.
7. **Make/miss labels:** every detected shot is labeled by hand in a labeling
   tool that plays the video with sound (the spoken "muz"/"elma" and the ball).
   A random 10% of shots, stratified by session, is labeled again blind. The
   disagreement rate is reported as the label error rate. Disagreements are
   reviewed.
8. **YOLO fine-tune:** validation set is a whole held-out video,
   never random frames from training videos.
9. **Experiment (step 9)** is the final out-of-sample test of A1.

Allowed inputs (known before release): release mechanics, shot number
within the session (warm-up / fatigue), shooter (A4 only).

## Fine-tuned ball model (decided before testing)

- Test: the release exam (67 manual shots, human labels) and the 16 reviewed
  shots. None of these shots were used in training.
- Success: within 2 frames about as often as yolo26m (67/67), reported with a
  95% interval and shot by shot; the same on the 64 blind labels only; a release
  for 0 of the 7 rebounds and gestures; at least 3x faster than yolo26m per frame
  on the same CPU.
- The release threshold is chosen again leave-one-session-out for the new model.
- Box metrics on S08 are secondary: those labels started from the teacher's
  boxes, and S08 was used to pick the best epoch.