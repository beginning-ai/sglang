---
name: jj-resolve
description: Resolve merge conflicts in Jujutsu (jj) repositories. Use when `jj` shows conflicted commits and you need a repeatable workflow to resolve each conflict commit by creating a temporary child commit, fixing files, reviewing the patch, and squashing the fix back into the original conflicted commit.
---

# JJ Resolve Conflicts

Resolve conflicts one conflicted commit at a time and keep history clean by squashing each fix back into the original conflicted change.

## Workflow

1. Inspect current conflict state.
- Run `jj log -r "conflicts()"`.
- Identify the first conflicted `change_id` to process.

2. Create an empty child commit on top of the conflicted change.
- Run `jj new <change_id>`.
- Use the printed file list as the authoritative set of files to resolve.

3. Fix conflicts in the listed files.
- Edit conflict markers and keep the intended final content.
- Run `jj resolve` when needed for file-level conflict resolution.

4. Check the final commit state.
- Run `jj show --git`.
- Confirm the resulting patch contains only the expected conflict-resolution changes.

5. Squash the fix back into the original conflicted commit.
- Run `jj squash`.

6. Repeat for the next conflicted commit.
- Run `jj log -r "conflicts()"` again.
- If conflicts remain, return to step 2 with the next `change_id`.

## Guardrails

- Use Jujutsu commands (`jj`) for the full workflow.
- Resolve one conflicted commit per loop to avoid mixing unrelated fixes.
- Re-run `jj show --git` before each `jj squash` to verify correctness.
