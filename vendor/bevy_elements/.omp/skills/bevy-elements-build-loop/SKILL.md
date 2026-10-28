---
name: bevy-elements-build-loop
description: "Run one iteration of the bevy_elements build loop: research one environment aspect in home-still, score candidates for game value, commit a research note, implement one improvement, inspect every visual element it creates on screen from several angles, distances and zooms, log it, then merge, push, tag and publish the crate. Use when asked to iterate, advance, or build out bevy_elements for games. Run from the repo root."
---

One run of this skill is one iteration. Each iteration researches one environment aspect in home-still, picks the feature with the highest game value, commits its research note, implements one improvement, verifies it on screen, logs it, then merges it to main, pushes, tags, and publishes the crate. State lives in `docs/loop-log.md`; rules live in `AGENTS.md`.

## 0. Read state

1. `AGENTS.md`: rules, traps, gates. Follow them; do not re-derive them.
2. `docs/loop-log.md`: the newest iteration's "Top 5 remaining items". They are candidates, alongside whatever research finds. The next iteration number is the last one plus one.
3. `docs/research/corpus.md`: the reading list per domain.
4. `git status --short`. The driver starts a pass only on `main` with no tracked changes, so the tree should be clean. Untracked `.omp/` and `scripts/` are the loop's own. If you find tracked changes you did not make, stop and log them; do not commit or revert them.

## 1. Preflight home-still

- Run `system_status` (MCP) or `hs status`.
- Distill must be healthy. If it is down, run `hs restart` once. If it stays down, skip `distill_search` and `distill_index`, and list each unindexed note under `Pending index`.
- State on 2026-10-16: distill (192.168.1.110:7434) is healthy, but `distill_index` fails on new documents with an ONNX out-of-memory error (about 2 GB allocation). Record affected stems under `Pending index`; do not retry in a loop.
- Scribe: 192.168.1.110:7435 is unavailable; bmb.local:7433 is healthy with no free slots. `scribe_convert` may queue or fail. If it does, rely on the corpus.
- Corpus: `~/.cache/bevy_elements_corpus/markdown` (828 MB). If it is missing, sync it per `docs/research/corpus.md`.

## 2. Research one aspect

Choose one aspect per iteration: sky and celestial, clouds, precipitation, fog, wind, lightning, terrain, erosion and hydrology, water, ground cover (snow and wetness), dust, voxel snow, or the flora boundary. Prefer the aspect with the highest open item in the loop log's backlog. Do not pick the aspect of the last two iterations unless the backlog demands it.

Search in this order:

1. Corpus, by method phrase: `grep -ril "<phrase>" ~/.cache/bevy_elements_corpus/markdown`. Use method terms (`Henyey`, `froxel`, `talus`, `horizon occlusion`). Topic words (`rain`, `snow`) drown in noise.
2. `distill_search` with `include_text=false` for semantic hits, then `markdown_read {stem, page}`.
3. A DOI not in the corpus: `paper_get`, then `paper_download`, then `scribe_convert` only if scribe is healthy. For paywalled PDFs, use skill `ingest-dois-via-anna-scidb`. For non-DOI sources, record the URL and read it.
4. Read at least two sources, one of them primary. Cite every stem in `## Sources`.

## 3. Choose the feature

Score every candidate you read, 1 to 5 on each axis:

| Axis | 5 means |
|---|---|
| Game hook | A gameplay system can read or react to it: a resource, a message, or a point query (wetness at a position, snow depth at a point, wind at a point, a strike's position). |
| Visible payoff | Changes what a player sees at 1080p in the capture harness. |
| Correctness | Fixes a physically wrong or missing result in shipped weather. |
| Cheapness | Small GPU cost, stated in the note and within the budgets in the README performance section; small code surface; no sibling change. |
| Ownership | Belongs to this crate under the ownership map in `AGENTS.md`. |

Priority is Game hook + Visible payoff + Correctness + Cheapness. Ownership is a gate: if another crate owns it, do not build it here; add a hook instead. Ties go to the higher Game hook.

Game hooks already exist. Start from the README's `Gameplay queries` section (`Environment` for weather, daylight and temperature; `wind_at`; `water_depth`; `water_velocity`; `snow_depth`; `wetness`), and from `src/lib.rs` and each plugin's public resources. Propose a gap in those hooks (missing query, stale or coarse answer, no way to read a state back) or a hook that does not exist yet, never a duplicate. Candidates to check: saving and restoring weather state, a deterministic seed for a whole environment, and a run path with no window.

Record the full score table in the log, not only the winner.

## 4. Research note before code

Create `docs/research/<system>.md` with these headings, in this order: `## Sources`, `## Model adopted`, `## Parameters`, `## GPU mapping`, `## Validation targets`, `## Deviations from this plan`, `## Pending index`. A default changes only with a cited source recorded under `Deviations`.

`Validation targets` gives one checkable target per inventory element of section 6: what the element must look like at 1×, in close-up, and from which angle.

Commit the note alone before writing code: `git add docs/research/<system>.md && git commit -m "Research: <system>"`.

## 5. Implement one improvement

- One improvement, within the note's `Model adopted`.
- Public surface: settings are `Resource + Default + Clone + Debug + Reflect`, added with `init_resource`; messages use `add_message`; everything is re-exported from the crate root (no `prelude`); plugins are `*Plugin`, system sets `*Systems`.
- Gameplay reads take `Option<Res<T>>`. A missing `Res<T>` panics its system.
- No `unwrap`/`expect` outside tests, no `unsafe`, no game vocabulary in `src/` (`tests/leaf.rs`), SI units at every boundary.
- One path per feature: no fallbacks, no shims. Delete the code the change replaces.
- Shaders are WESL. Rebuild the examples after any shader edit, or captures silently use the old shader.
- Tests: for a bug, a test that fails before and passes after. Permanent tests check consumer-visible behavior, boundaries and invariants. Do not test wiring, defaults, source text or mock echoes; use throwaway checks in `/tmp` for those.

## 6. Inspect and verify

Run every gate before logging:

1. `cargo build --release && cargo test && cargo build --examples --all-features`
2. `cargo clippy --all-targets --all-features -- -D warnings`
3. `RUSTDOCFLAGS=-D warnings cargo doc --no-deps --all-features`
4. `cargo build --lib --target wasm32-unknown-unknown`
5. GPU tests, locally only, one process per module:
   - `cargo test --release --tests -- --ignored --test-threads=1 --skip ::gpu_tests::`
   - `for m in clouds terrain cover dust; do cargo test --release --lib --features=isomesh -- --ignored --test-threads=1 $m::gpu_tests::; done`
   - `clouds::gpu_tests` has eleven tests and exhausts the adapter on the eleventh. Run it with `--skip the_horizon_sky`, then `the_horizon_sky` alone.
   - Three failures are pre-existing on HEAD: `cover::gpu_wetness_soaks_and_dries_on_the_formula`, `cover::gpu_surface_weathering_follows_the_map`, `voxel::gpu_voxel_snow_covers_the_top`. Confirm against `/tmp/base` before blaming your diff.
   - Offload gates 1 to 4 to `sib` or `bmb` only when the local CPU is the bottleneck. Never offload GPU tests.
6. Build an inventory before shooting. List every visual element the change creates or alters: each sky body, cloud layer, fog volume, precipitation type, surface effect, shadow, horizon edge, puddle, flood, dust front, and each object in the examples that the change touches. Every inventory element gets its own shots. An element you cannot find on screen is a finding, not a skip.
7. Shoot each inventory element as follows. Keep the captures under `/tmp/iN/`.
   - Full frames at 1920×1080 for every weather preset and time of day the element changes (`ELEMENTS_PRESET`, `ELEMENTS_HOUR`, `ELEMENTS_GROW_HOURS`). Keep the status band in full frames; it names the scene's weather state. Full-frame command:
   ```sh
   xvfb-run -a -s "-screen 0 1920x1080x24" env ELEMENTS_FRAMES=300 ELEMENTS_SHOT=/tmp/iN/x.png \
     ELEMENTS_PRESET=Snow ELEMENTS_HOUR=12 ./target/release/examples/<example>
   ```
   - At least two camera angles that differ in pitch or heading: one grazing the surface, one from above or far away when the element is visible at distance.
   - Crops at 2× and 4× of every place the element meets geometry, a horizon, a shadow edge, or another effect: `magick in.png -crop WxH+X+Y +repage -filter point -resize 200% out.png`. Judge horizons only below the top 56 rows.
   - Motion: a frame sequence (`ELEMENTS_FRAMES`, `ELEMENTS_SHOT_DIR`) for anything that moves, streaks, flickers or accumulates, assembled into a GIF or contact sheet with `ffmpeg`.
   - Before and after: the same view from `/tmp/base` (`git worktree add /tmp/base HEAD`) beside the new one, so the change is visible, not only the result.
8. Send every shot to the `screenshot-reviewer` agent on Gemini Flash only (`openrouter/google/gemini-3.8-flash`), in batches of at most four, as its own process on Chat Completions (`PI_OPENROUTER_RESPONSES=0`); the skill `bevy-elements-capture-harness` gives the exact command. For each shot, give the reviewer the inventory element it belongs to and its target from the note's `Validation targets`. Ask for a verdict per shot: pass, or defect with file, region, severity and evidence. A shot without a verdict has not been reviewed. Do not judge pixels yourself.
9. Fix and re-shoot. Every defect goes back to the code, then the examples are rebuilt (shaders included), then the affected elements are re-shot and re-reviewed. Repeat until each inventory element passes at 1× and in close-up. A defect that shows at 1× blocks shipping. Log it as open, with severity and reason, only if it cannot be fixed in this iteration.
10. Performance, second priority. If the GPU path changed, report the minimum over the pass's `Diagnostic::values()`, and log `nvidia-smi --query-gpu=pstate,clocks.gr --format=csv,noheader -lms 1000` in the same shell. Means under Xvfb read 5 to 10 times slow.

## 7. Document, ship and publish

Start only after section 6 passes and every inventory element is reviewed. Each step below stops the pass on failure: log the exact error, leave the repo as it is, and exit. Never publish a version whose gates are red.

1. Document. Fill in the note's `Deviations` and `Pending index`. Add this iteration's changes to `## [Unreleased]` in `CHANGELOG.md`, under the Keep a Changelog headings (`Added`, `Changed`, `Fixed`, `Removed`). Update README only for user-visible changes.
2. Version. Bump the patch version in `Cargo.toml` (0.5.0 to 0.5.1). Move the Unreleased entries to `## [X.Y.Z] — <date>` and leave an empty `## [Unreleased]`. The crate is pre-1.0, so a release may break the API; say so in the entry when it does.
3. Log. Append `## Iteration N — <date> — <title>` to `docs/loop-log.md`: what changed and why it ranked first; the candidate score table; the shot list (every file path, its inventory element, and the reviewer's verdict); defects found, fixed and re-shot; defects still open with severity; verification (commands and results); the top 5 remaining items, ranked.
4. Branch and commit. `git checkout -b loop/iter-N` from `main`. Commit by path, in this order: the research note (already committed in section 4), the implementation, the version and changelog bump, then the log (`Loop log: iteration N, <title>`). `git status --short` must be empty afterwards.
5. Push the branch: `git push -u origin loop/iter-N`.
6. Merge: `git checkout main && git pull --ff-only && git merge --no-ff loop/iter-N -m "Merge loop/iter-N: <title>"`. On a conflict, run `git merge --abort`, log it, and stop.
7. Re-verify on main: run section 6 gates 1 to 4 on the merged tree. A failure stops the pass before the push.
8. Push main: `git push origin main`, then `git pull --ff-only` so local main matches origin.
9. Tag: `git tag -a vX.Y.Z -m "bevy_elements X.Y.Z"`, then `git push origin vX.Y.Z`.
10. Publish: `cargo publish --dry-run`, then `cargo publish`. Log the crates.io URL. A published version cannot be replaced, only yanked, so step 7 is the last gate.

## 8. Stop

One iteration per run. Stop when the iteration is shipped: note committed, feature built, every inventory element reviewed with a passing verdict or an open defect logged, gates green or pre-existing failures named, the branch merged to main and pushed, the tag pushed, the crate published at the new version, and the log committed. Stop early only if the user says stop, or a required tool stays unreachable after the checks above (name the check). A failed gate is not a blocker: fix it or report it. A visible defect is not a reason to stop: fix it and re-shoot. A failed push, merge or publish is a blocker: log the exact error and stop; do not force-push, and do not retry a publish.
