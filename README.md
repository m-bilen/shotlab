# Swish or Brick
What decides a free throw? Pose estimation and ball tracking on 2,000 of my own shots.

![tests](https://github.com/m-bilen/swish-or-brick/actions/workflows/tests.yml/badge.svg)

Analyzing my own basketball shots with pose estimation and machine learning
to find which body mechanics are linked to made shots.

## Status

Work in progress, target: January 2027.

- Done: 2,000 shots recorded, every shot and its release moment found automatically
  (release within 2 frames of hand labels on all 103 unseen test shots).
- Next: make/miss labels, body mechanics at release, analysis.

- [Roadmap](docs/roadmap.md): what is done and what is next
- [Analysis plan](docs/analysis_plan.md): questions and tests, written before the results
- [Decisions](docs/decision.md): every choice and mistake, with the reason