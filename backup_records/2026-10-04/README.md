# Backup and recovery record — 2026-10-04

Repository: yunseo326/3robot_project

Branch: `backup/2026-10-04`

Selected source files: 75

## Restore

MuJoCo biped simulation. STAGE 2 is recorded as passed; STAGE 3 awaits user decision. Read logs/progress.md and docs/robot_spec.md. Restore dependencies from environment.yml. Models/envs/scripts/reference motions and progress are preserved; local training checkpoints, new videos, tensorboard data, frame dumps and Blender binary sessions are not newly uploaded. Existing remote media are inherited. Re-training is needed for excluded policies; full trained-state recovery is not claimed.

## Verification and limits

Files were read without executing project code. Selected text and extracted Office/PDF text were checked using Gitleaks v8.30.1, known credential-value matching, and suspicious-code patterns. Findings in experiment comparison/input hashes were reviewed as non-credentials. Binary files listed in summary.json were not covered by text secret scanning. No whole-computer malware clearance or full-history audit is claimed.

`manifest.json` records exact source and uploaded SHA-256 plus Git blob IDs. `exclusions.csv` records excluded paths/reasons; directory entries represent excluded subtrees and are not file counts. Existing remote files outside selected scope may be inherited, but no local unpushed history is added. No raw secrets are recorded here.

This record describes a selected GitHub archive, not a USB or retained local backup. Original source folders are unchanged.
