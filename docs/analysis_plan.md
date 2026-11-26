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
- About 1,000 real shots per shooter (2,033 releases found).

## 2. Primary features (frozen before any make/miss label is seen)

All measured at release or before it, from the right side, in body units where
a length is involved:

1. Elbow angle at release (shoulder-elbow-wrist)
2. Upper arm angle at release (hip-shoulder-elbow)
3. Knee bend: smallest knee angle (hip-knee-ankle) between shot start and release
4. Trunk lean at release (shoulder-hip line vs vertical)
5. Release height: ball center height above the ankles, in torso lengths
6. Wrist speed in the last 50 ms before release, in torso lengths per second
7. Time from shot start to release

Left out on purpose: ball launch angle and speed (measured after release, rule
2); elbow flare (not visible from the side); feet and gaze (not measured
reliably). Anything else is exploratory and reported as such.

## 3. Analyses

**A1. Mehmet's best form** (main result)
- Data: Mehmet's shots only.
- A1a. Together: a model with all primary features, leave-one-session-out.
  Does release mechanics predict makes better than the base rate? (ROC AUC)
- A1b. One by one, best range: for each primary feature, split the training
  sessions into 5 equal-count bins (edges from training only), pick the bin
  with the highest make rate, and on the held-out session compare the make
  rate in that bin with the shooter's make rate in that session. Differences
  are pooled over all held-out sessions.
- Significance: shuffle make/miss within each session (10,000 times) and
  repeat the whole procedure. Holm correction over the primary features.
- Sensitivity: A1 is also run with features centered within each session.
  Ranges that hold only in raw values are reported as such.

**A2. Father's best form**
- Same as A1 on father's shots.

**A3. Comparison**
- Are the best bins from A1b and A2b different? Compared in each shooter's own
  scale (session-centered), since raw angles also depend on camera angle.

**A4. Consistency** (pooled, both shooters)
- Question: does deviating from your own normal form lower the make rate?
- For each primary feature: z = (x - median) / MAD, with median and MAD from
  the same shooter's shots in the same session (no labels used). This removes
  day-level offsets such as camera angle.
- Primary measure: the mean of |z| over the primary features (how unusual the
  shot is overall). Deviations in either direction count, so they cannot cancel.
- Secondary: |z| per feature; and median and MAD from the training sessions
  instead, which keeps day-level shifts in form but mixes them with camera angle.
- Exploratory: signed z, and z with z^2, to see whether the effect is one-sided.
- Shooter is a model input; raw (not normalized) features are not.

A5 (hot hand) is below.

## 4. Evaluation

- Split: leave-one-session-out. All shots of the held-out session,
  from both shooters, are excluded from training.
- Models: logistic regression, random forest.
- Baselines: always predict the majority class; shooter's base make rate.
- Metrics: ROC AUC, log loss, accuracy.
- Report results separately for 30 fps and 60 fps sessions.
- Expected effect is small: a free throw without a jump is a consistent motion.
  A null result is reported as such, together with the smallest effect we could
  have detected. A4 (consistency) is the most likely place for a signal.
- Camera geometry: per session, the shoulder separation in the side stance at
  the start of the recording (relative to torso length) is reported as a check
  of viewing angle.

## 5. Power and multiple tests

Smallest effect detectable with 80% power, assuming ~1,000 shots per shooter
and a 65% make rate (recomputed with the real make rate after labeling; this
uses no feature-outcome information):
- A1b/A2b: best bin (~200 shots) vs the rest: ~11 points for one feature,
  ~13-15 points with Holm correction over the primary features.
- A4: ~3 points per 1 SD of mean |z| (both shooters, one continuous measure).
- A5: make rate after a make vs after a miss: ~9 points.

Corrections: Holm over the primary features in A1b/A2b. A4 and A5 have a single
primary test and no correction. Secondary and exploratory results are not
corrected and are labeled as such.

## 6. Leakage rules

1. **Split by session, never by random shots.** Shots in one session share
   light, camera and fatigue.
2. **Only movement up to the release moment is a feature.** Nothing after
   release: the shooter reacts to the outcome. `duration_s` from shot
   detection is NOT a feature (it runs past release); time from shot start
   to release is.
3. **Every fitted step uses training sessions only:** scaling, feature
   selection, "best range" bins, thresholds. One planned exception: in A4 and
   the session-centered sensitivity runs, median and MAD come from the
   held-out session's own features, never its labels.
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
## Full-run release check (decided before looking)

- 40 random released shots (20 per shooter), outside the exam and the audit,
  labeled blind by the written rule.
- Accept if at least 36 of 40 are within 2 frames and at least 17 of 20 per
  shooter. Otherwise look for a pattern; any fix is checked on a new random
  sample, not on these 40.

## A5: Hot hand (decided before make/miss labels exist)

- Question: is a shot more likely to go in after a make than after a miss?
- Main measure (per shooter): P(make | previous make) - P(make | previous miss),
  within sessions, using consecutive real shots only.
- Test: permutation test that shuffles makes and misses within blocks of 20
  shots in each session, 10,000 times. This keeps slow trends (warm-up, fatigue)
  and accounts for the streak selection bias (Miller and Sanjurjo, 2018).
- Secondary: after 2 and 3 makes in a row vs after 2 and 3 misses (fewer shots,
  weaker test).
- Rhythm: compare the time to the next shot after makes and after misses. Then
  the hot hand difference with time to next shot controlled, and, after misses
  only, whether a short wait predicts a make. These show whether an effect could
  come from rhythm rather than confidence; they are not proof of a cause.
- Power: see "Power and multiple tests". A null result is reported with that
  limit.