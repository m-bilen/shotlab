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
| S05 | 2026-11-14 | ~7 m | 30 | electric | ~150 | ~195 (2 videos) |
| S06 | 2026-11-14 | ~7 m | 30 | electric | ~150 | ~150 |
| S07 | 2026-11-15 | ~7 m | 60 | planned | ~150 | ~150 |

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
6. **Release detection:** tuned on S02 labels, error reported on other sessions.
7. **Audio label model:** trained on S01-S04, tested on S05-S07
   (random sample checked by hand). Model labels are never used to
   evaluate the model itself.
8. **YOLO fine-tune:** validation set is a whole held-out video,
   never random frames from training videos.
9. **Experiment (step 9)** is the final out-of-sample test of A1.

Allowed inputs (known before release): release mechanics, shot number
within the session (warm-up / fatigue), shooter (A4 only).