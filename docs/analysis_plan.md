# Analysis plan

Single source of truth for data, features, tests and leakage rules.
Frozen as git tag `prereg-v1` before any make/miss label existed. Any later
change is listed under "Deviations" at the end, with date and reason.

## 1. Data

| Session | Date | Camera | FPS | Light | Mehmet | Father |
|---|---|---|---|---|---|---|
| S01 | 2026-11-11 | 9 m | 60 | electric | 154 | 150 |
| S02 | 2026-11-11 | 6 m | 60 | electric | 50 | 49 |
| S03 | 2026-11-12 | 7.5 m | 60 | daylight | 151 | 148 |
| S04 | 2026-11-12 | 6.5 m | 60 | dark | 102 | 97 |
| S05 | 2026-11-14 | 7 m | 30 | electric | 152 | 188 (2 videos) |
| S06 | 2026-11-14 | 7 m | 30 | electric | 155 | 148 |
| S07 | 2026-11-15 | 7 m | 60 | electric | 153 | 142 |
| S08 | 2026-11-15 | 7 m | 60 | electric | 100 | 94 |

Counts are detected shots with a release (2,033 in all). A few are rebounds;
labeling marks them "not a shot".

- Unit of analysis: one shot.
- Shot type: free throw without a jump, same spot, camera on the side.
- Label: made or missed, by hand (section 7).
- Release time: from ball-to-fingertip distance with the fine-tuned yolo26n,
  threshold 1.15 ball sizes. Within 2 frames of hand labels on 103 of 103
  unseen test shots.
- Shot number: order of a shooter's shots in a session, by time. The father's
  two S05 videos count as one session; the second continues the count.

## 2. Primary features (frozen)

### Definitions
- **Shot start:** when the shot detector fires (right wrist above nose). Not a
  body event; used only to place windows.
- **Release:** the release time of the shot (between frames).
- **Torso length:** median right shoulder to right hip distance over all
  frames of that shooter in that session.
- **Landmarks:** right shoulder, elbow, wrist, index finger, hip, knee, ankle
  from MediaPipe. A landmark with visibility below 0.5 is missing.
- **Ball:** from the fine-tuned yolo26n. If no ball is found within one frame
  of the release, release height is missing for that shot.
- **Real shots:** detected shots not labeled "not a shot". Normalizations use
  all real shots (made, missed and can't tell alike); they use no outcome.

### Signal processing (fixed)
1. Landmark positions are interpolated from their real timestamps onto an even
   time grid at the video's nominal rate (16.7 ms at 60 fps, 33.3 ms at 30 fps),
   because phone videos drop frames (most in S03, S04 and S07). Gaps longer
   than 100 ms are not filled.
2. The grid is smoothed with a Savitzky-Golay filter over ~150 ms (9 samples
   at 60 fps, 5 at 30 fps), polynomial order 2.
3. Values "at release" are interpolated at the release time.
4. A feature is missing for a shot if its window has an unfilled gap or a
   missing landmark. Missing counts are reported per feature and session.

### Features
1. **Elbow angle** at release: shoulder-elbow-wrist.
2. **Upper arm angle** at release: hip-shoulder-elbow.
3. **Knee bend:** knee angle (hip-knee-ankle) at the last local minimum before
   release that is at least 5 degrees below the knee angle at release, searched
   from W = 1.0 s before shot start to release; if there is none, the smallest
   angle in that window. Checked before the freeze on 16 shots per shooter,
   without labels (decision.md).
4. **Trunk lean** at release: angle of the shoulder-hip line to the image
   vertical, positive when the shoulder is ahead of the hip toward the basket.
5. **Release height:** ball center height above the ankle at release, in torso
   lengths.
6. **Wrist speed:** smoothed wrist displacement over the last 100 ms before
   release, divided by 0.1 s, in torso lengths per second.
7. **Knee-to-release time:** from the moment of feature 3 to release, in ms.

Left out on purpose: ball launch angle and speed (after release, rule 2);
elbow flare (not visible from the side); feet and gaze (not measured
reliably). Anything else is exploratory and labeled so.

### Reliability (measured without labels)
- Timing: the change in each feature when the release moves by 1 frame,
  compared with its shot-to-shot spread.
- Jitter: the spread of each angle and of wrist speed during the 3 s still
  side stance at the start of each recording. A lower bound for noise during
  motion.
- Reported per feature and per fps. Noisy features pull A4 toward no effect.

### Two kinds of normalization
- **Session-centered:** value minus the median of that feature over both
  shooters' shots in the session. Both shooters were filmed by the same camera
  from the same spot, so this removes camera angle but keeps the difference
  between the two shooters.
- **Shooter-session z:** (value - median) / (1.4826 x MAD) over that shooter's
  shots in that session. Removes camera angle and that day's form shift.
- Neither uses labels.

## 3. Analyses

All analyses use real shots labeled made or missed. "Can't tell" shots are
left out and checked separately (section 7).

### A1. Mehmet's best form (main result)
- Data: Mehmet's shots.
- **A1a. Together:** logistic regression with the 7 features and shot number,
  leave-one-session-out, scaling fitted on training sessions. Settings fixed
  now: scikit-learn defaults (L2, C = 1.0); random forest with 500 trees,
  at least 20 shots per leaf, seed 0. No tuning. Shots with any
  missing feature are left out (count reported). Metric: ROC AUC within each
  held-out session, averaged weighted by shots, so that differences between
  sessions cannot raise it. Baseline: logistic regression with shot number
  only. Statistic: AUC(mechanics) - AUC(baseline). Test: one-sided, make/miss
  shuffled within blocks of 20 consecutive shots in each session (10,000
  times), both models refitted each time. Random forest, pooled AUC and log
  loss are secondary, without a test.
- **A1b. One by one, best range:** for each feature, in the training sessions
  split it into 5 equal-count bins (edges from training only) and pick the bin
  with the highest make rate (ties: the bin closer to the median). In the
  held-out session, compare the make rate inside that bin with the make rate
  of the shooter's other shots. Pool over held-out sessions: make rate of all
  shots that fell in their fold's chosen bin minus that of all other shots.
  Test: one-sided, block shuffle as in A1a, whole procedure repeated. Holm
  correction over the 7 features. Also reported: in how many held-out sessions
  the bin was better, and the bin edges from a fit on all sessions.
- **Sensitivity:** A1a and A1b repeated with session-centered features.
- **A range "holds"** if it passes Holm in raw values and its one-sided p is
  below 0.05 in the session-centered run.

### A2. Father's best form
- Same as A1 on the father's shots.

### A3. Comparison (descriptive, no test)
- For each feature: the median difference between the two shooters within each
  session, averaged over sessions, with a 95% interval from resampling sessions.
  Camera angle cancels out because both were filmed together.
- For features where a range holds for at least one shooter: both shooters'
  best bins in session-centered values, with intervals. If no range holds,
  this part reports that there is nothing to compare.

### A4. Consistency (pooled, both shooters)
- Question: does deviating from your own normal form lower the make rate?
- For each feature: shooter-session z. Primary measure: mean |z| over the 7
  features (how unusual the shot is overall). Deviations in either direction
  count, so they cannot cancel.
- Primary test: logistic regression make ~ mean |z| + shot number + one term
  per session and shooter (each shooter's make level that day). Statistic: the
  coefficient of mean |z|. Two-sided p from shuffling mean |z| within blocks of
  20 consecutive shots of the same shooter in each session (10,000 times).
  Effect reported as change in make probability per 1 SD of mean |z|, with a
  95% interval from resampling blocks.
- This is an inference model on all sessions, not a prediction of held-out
  sessions, so the session-shooter terms do not break rule 4.
- Secondary: |z| per feature; z from the median and MAD of the shooter's other sessions instead
  (keeps day-level form shifts, mixed with camera angle); Mahalanobis distance
  from the shooter-session typical shot with robust covariance (does not count
  the correlated arm features three times).
- Exploratory: signed z, and z with z^2, to see whether the effect is one-sided.
- Limitation: mean |z| gives the three arm features (1, 2, 5) more weight
  because they move together, and noisy features pull it toward no effect.

### A5. Hot hand (per shooter)
- Question: is a shot more likely to go in after a make than after a miss?
- Pairs of consecutive real shots in a session. "Not a shot" detections are
  skipped; a "can't tell" breaks the sequence.
- Statistic: P(make | previous make) - P(make | previous miss).
- Test: two-sided, make/miss shuffled within blocks of 20 consecutive shots in
  each session (10,000 times). Keeps slow trends (warm-up, fatigue) and accounts
  for the streak selection bias (Miller and Sanjurjo, 2018).
- Secondary: after 2 and after 3 makes in a row vs the same number of misses.
- Rhythm (secondary, not proof of a cause): time to the next shot after makes
  vs after misses; the hot hand difference with time to the next shot as a
  covariate; after misses only, whether a short wait predicts a make.
  
### A6. Warm-up and fatigue (secondary)
- Make rate over shot number: logistic regression make ~ shot number + one term
  per session, per shooter. Two-sided p from shuffling make/miss within each
  session (10,000 times).
- Features over shot number: the median of each feature in the first and last
  quarter of each session, averaged over sessions. Descriptive.

### M. Cost of leakage (method, decided before results)
These analyses use a wrong setup on purpose, to show how much it would inflate
the results. They are never reported as findings about shooting.
- **M1. Random split:** A1a and A2a repeated with a random 5-fold split of
  shots instead of leave-one-session-out, pooled AUC. Compared with the main
  result.
- **M2. After-release information:** A1a and A2a repeated with the ball's
  launch angle (first 3 ball positions after release) added as a feature. Run
  only if at least 80% of shots have the ball in 3 frames after release;
  otherwise reported as not possible.
- **M3. Random frames for the ball model (optional, after January):** a second
  student trained with S08 frames mixed into training, tested on the same
  exam, to show what a random frame split would have hidden.

## 4. Evaluation and reporting

- Leave-one-session-out over the 8 sessions: all shots of the held-out session,
  from both shooters, are out of training.
- A1a and A4 are also reported separately for 30 fps (S05, S06) and 60 fps.
- Camera angle check: shoulder separation in the side stance at the start of
  each recording, relative to torso length, per session.
- Expected effects are small: a free throw without a jump is a consistent
  motion. A null result is reported with the smallest effect we could detect.
- Every p-value names its test; effects come with 95% intervals.
- Random seed 0 for every shuffle and resample.

## 5. Power and multiple tests

Smallest effect detectable with 80% power, assuming ~1,000 shots per shooter
and a 65% make rate. Recomputed with the real make rate after labeling (this
uses no feature-outcome information).
- A1b/A2b, best bin (~200 shots) vs the rest, one-sided: ~10 points for one
  feature, ~12-14 points with Holm.
- A4: ~3 points per 1 SD of mean |z|.
- A5: ~9 points.
- A6 and M1-M3 are secondary; no correction.

Holm over the 7 features in A1b, separately for each shooter. A1a, A4 and A5
have a single primary test and no correction. Secondary and exploratory
results are not corrected and are labeled so.

## 6. Leakage rules

1. **Split by session, never by random shots.** Shots in a session share light,
   camera and fatigue. M1 breaks this on purpose, only to show its cost.
2. **Only movement up to release is a feature.** Frames after release are used
   only to find the release moment. The detector's `duration_s` is not a
   feature (it runs past release). M2 breaks this on purpose, only to show its
   cost.
3. **Fitted steps use training sessions only:** scaling, bins, models.
   Planned exceptions that use held-out features but never labels:
   session-centered values, shooter-session z, Mahalanobis distance.
4. **No session-identifying inputs in predictive models:** camera distance,
   fps, date, light. Used only to report results per group.
5. **Each shot once:** rebounds and gestures are removed by "not a shot".
6. **Release detection:** one threshold, chosen leave-one-session-out on manual
   release frames. The search window (detection start -200 ms to end +200 ms)
   was set after looking at all manual frames.
7. **YOLO fine-tune:** validation on a whole held-out video, never random
   frames from training videos.
8. **Check samples:** any sample for a check excludes every shot used in
   training, tuning or an earlier check, and this is done in code.

Allowed inputs: the 7 features; shot number (A1a, A4, A6); time to the next
shot (A5 rhythm only); shooter and session-shooter terms (A4, A6).

## 7. Make/miss labels

- Tool: plays each shot with sound. Keys: made, missed, not a shot, can't tell.
- The spoken word ("muz"/"elma") decides. If unclear, the ball at the rim
  decides. "Can't tell" only when neither does.
- Labeling in sittings of at most ~400 shots.
- Both checks are done after all first labels are finished and at least one
  day later, so first decisions are not remembered.
- Check (a): a random 10% of shots, stratified by session and shooter, labeled
  again with sound, without seeing the first labels. Measures labeling slips.
- Check (b): 100 other random shots (50 per shooter), labeled with the sound
  off from the ball at the rim only, without seeing other labels. Also catches
  a wrong word said at recording time.
- Disagreements are reviewed with sound and classified: wrong key, wrong word,
  truly unclear. Corrected labels are used. Error rates reported with intervals.
- "Can't tell": counted per session and shooter. Every primary result is
  repeated with all of them as makes and again as misses.

## 8. Experiment (A-B-B-A), conditional

- Run only if a range holds in A1b or A2b (section 3), and only for that
  shooter.
- Blocks of 25 shots, alternating "aim for the range" and "normal shot" in
  A-B-B-A order to balance warm-up and fatigue. At least 400 shots per
  condition (800 in all, 32 blocks), for ~9 points of power.
- The target range is frozen in a new tag before shooting.
- Manipulation check: the feature is measured on every shot; the share inside
  the range is reported per condition. If "aim" does not clearly raise it, the
  experiment tested nothing and is reported so.
- Primary test: make rate difference between conditions, condition labels
  shuffled between blocks (not between shots).
- Interpretation: the effect of the instruction, which may also change
  attention or rhythm, not of the range alone.
- A smaller experiment would not be informative and is not run. If no range
  holds, it is not run and this is reported.

## 9. Completed checks (decided before looking)

**Fine-tuned ball model.** Test: the release exam (67 manual shots) and the 16
reviewed shots, none used in training. Success: within 2 frames about as often
as yolo26m (67/67), with a 95% interval; the same on the 64 blind labels; a
release for 0 of 7 rebounds and gestures; at least 3x faster per frame on the
same CPU. Threshold chosen again leave-one-session-out. Box metrics on S08 are
secondary (labels started from the teacher's boxes; S08 picked the best epoch).
Result: passed (decision.md).

**Full-run release check.** 40 random released shots (20 per shooter), outside
the exam and the audit, labeled blind. Accept if at least 36 of 40 within 2
frames and at least 17 of 20 per shooter. Result: 4 of the 40 turned out to be
in the training set; 36/36 of the rest within 2 frames (decision.md).

## Deviations

None yet.