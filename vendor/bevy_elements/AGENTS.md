# bevy_elements — notes for agents

Living environments for Bevy: celestial bodies, climate-driven weather, GPU wind, clouds, precipitation, fog, lightning, heightfield terrain with erosion and flooding, and ground cover, simulated in compute shaders. Single crate; each domain is a module with one plugin.

## Version pin

Pinned to **Bevy 0.20.0**. Read the vendored source and migration guides, not bevy.org (which tracks `main`): `~/.cargo/registry/src/index.crates.io-*/bevy-0.20.0/` with `examples/` and `_release-content/migration-guides/`. The verified 0.20 render API paths are in `docs/research/bevy-0.20-render.md`.

## Source of truth

This repository, [`Ladvien/bevy_elements`](https://github.com/Ladvien/bevy_elements), is canonical. MIT OR Apache-2.0. Consumers (Sanctuary at `/Users/ladvien/sanctuary` on the Mac) get fixes here, in their own commits with tests, and never patch a copy.

## Research first

No system is implemented without its `docs/research/<system>.md`, committed first. Protocol:

1. **Search the corpus first.** The library already holds 12 178 converted documents, many under non-DOI stems (SIGGRAPH *Advances* course notes as `s<year>-advances-*`, books as `book_*`). `docs/research/corpus.md` lists what is there per domain and how to sync and search it (`~/.cache/bevy_elements_corpus/markdown`, phrase search, not catalog titles). Every note cites the relevant corpus documents, not only the plan's seeds.
2. Preflight with `hs status` (MCP `system_status`). If distill is down, run `hs restart` once; if it stays down, research continues without `distill_search` and `distill_index` is skipped and listed under the note's `Pending index`.
3. For each seed DOI not in the corpus: `hs paper get --doi <doi>`, then `hs paper download --doi <doi>` (MCP `paper_download`). Paywalled: skill `ingest-dois-via-anna-scidb`. Non-DOI sources: `hs paper search "<title>"`, else record the URL and read it.
4. Convert with `hs scribe convert <pdf> --out <md>` (MCP `scribe_convert {stem}`; stem = lowercased DOI, `/` → `_`). If scribe drops regions, read the PDF's text directly and say so in the note.
5. Read with MCP `markdown_read {stem, page}` or the synced file; index with `distill_index {stem}` when healthy.
6. Write `docs/research/<system>.md` with headings: `## Sources`, `## Model adopted`, `## Parameters`, `## GPU mapping`, `## Validation targets`, `## Deviations from this plan`, `## Pending index`. A default changes only with a cited source recorded under `Deviations`.

## Plays well with the family

1. **Bevy pin parity.** Depend only on crates pinned to the same Bevy minor. Couple to 0.19 crates (`bevy_carnage`, `bevy_light_grid`, `bevy_stigmergy`) only through plain data in game glue, documented in the README `Plays well with` section.
2. **Ownership map.** Foliage, trees, tree wind sway, leaf colour and snow on branches belong to `bevy_flora`. Isosurface and voxel meshing belongs to `isomesh`/`bevy_isomesh`. Blood wetness belongs to `bevy_carnage`. Creature light grids belong to `bevy_light_grid`. WFC and authored world layout belong to `bevy_emerge`. This crate owns sky, celestial bodies, weather, atmosphere-driving, wind fields, precipitation, clouds, fog, lightning, heightfield terrain and its simulation, ground cover, and the snow slab over a voxel volume.
3. **Hooks over dependencies.** This crate never makes a sibling depend on it. It feeds siblings through their existing public resources behind an optional feature (`flora`, `isomesh`). Hook data is published as plain resources and messages (`WindState`, `WindGust`, `SeasonState`, `SurfaceWeathering`, `Environment`) that any crate or game can read.
4. **A needed sibling change lands in the sibling's repo**, in its own commit with tests there, and is published before this crate depends on it.
5. **Units are SI** at every boundary (m, s, m/s, °C, lux, mm/h). Wind direction is the unit XZ vector the wind blows *toward*, which matches `FloraWind::direction`.
6. **No game vocabulary** in `src/`, enforced by `tests/leaf.rs`.

## Rules

- **One plugin per domain; split by domain, not by kind.** Settings are `Resource + Default + Clone + Debug + Reflect` added with `init_resource`, so a caller who inserts one first overrides it. Messages use `add_message`. No `prelude`: everything is re-exported flat from the crate root. Plugins are `*Plugin`, system sets `*Systems`.
- **No `unwrap()`/`expect()` outside tests.** One path per feature: no fallbacks, shims or stubs.
- **A missing `Res<T>` panics its system.** Take `Option<Res<T>>` or `init_resource`. Run conditions are all evaluated, with no short-circuit.
- **`Single<..>` silently skips its system on a non-unique match.**
- **More than ten GPU tests in one process exhaust the adapter.** Each test builds a Bevy app and wgpu instance; on the eleventh `request_adapter` returns `None` and the test panics with `GPU_NOT_FOUND` at `bevy_render-0.20.0/src/renderer/mod.rs:308`, while the same test passes alone. `clouds::gpu_tests` has eleven: run the suite with `--skip the_horizon_sky`, then the skipped test on its own.
- **The examples' harness burns a status line into the top ~56 rows** of every capture (a semi-transparent band over the sky). A reviewer may read it as an atmospheric shelf; crop below it when judging horizons, and read it for the weather state at the capture (preset, cover, visibility).
- **GPU determinism.** GPU results are not bit-deterministic. Astronomy, the weather scheduler and lightning are deterministic and seeded; goldens are named `the_<thing>_is_frozen`.
- **WebGPU limits.** Storage textures are write-only `rgba16float`/`rgba8unorm`, at most 4 per stage; ping-pong pairs, no read_write textures. Simulation state lives in storage buffers.
- Close any Bevy app you launch for a test.
- Do not hard-wrap markdown.
- **Never judge an image yourself.** Delegate visual judgment to the `screenshot-reviewer` agent (the `@vision` model reads the image; a text model must not look at pixels).

## Bevy 0.20 traps found here

- **A camera needs a render target.** Bevy's atmosphere environment probe unwraps the view uniforms, so an `ElementsCamera` with no window or image target panics the render world. GPU tests render into an `Image` target (`tests/support`).
- **Pipelined rendering moves the render app away.** After `App::finish()` the `RenderApp` sub-app is gone from the `App` (it runs on its own thread). Register render-world systems and resources before `finish()`, and report back through an `Arc<Mutex<…>>` resource. Wait for `PluginsState::Ready` (ticking the task pools) before `finish()`: the renderer is created asynchronously.
- **Pipelines compile asynchronously.** A kernel that is not ready yet skips the frame; anything read back meanwhile is zeros. GPU state is published as live (`WindGpu::live`) only after the first real dispatch. Tests use `RenderPlugin { synchronous_pipeline_compilation: true }`.
- **`FogVolume`s are extracted regardless of `Visibility`.** Switch one off with `density_factor = 0`.
- **`FogVolume` light attenuation uses the bounding-sphere radius.** Every sample's light is attenuated by `exp(−density · R)`: keep volumes small or compensate with `light_intensity` (`fog::volume_light_compensation`).
- **Bevy's atmosphere is lit by every `DirectionalLight`'s `color · illuminance`.** Dimming the sun light for cloud shadow also darkens the sky, the aerial perspective and the environment map; there is no per-light opt-out, so a fill light would scatter in the sky too. The cloud composite restores the sky with `sky_gain` (`src/clouds/composite.wesl`).
- **`AmbientLight` diffuse is shaded by `F_AB(1, NdotV).x` ≈ 0.452** (`pbr_ambient.wesl:18`): divide the wanted radiance by it. It is uniform over normals, so it cannot shade tops brighter than sides.
- **`DistanceFog::color` is added after exposure**, so it is a display-referred value: scale it by the camera's `Exposure::exposure()`.
- **Bevy's default `GlobalAmbientLight` is a constant 80 cd/m²**, which a night exposure lifts to white; `ElementsCamera` zeroes the per-camera `AmbientLight`.
- **Shaders are WESL.** `embedded://bevy_elements/noise.wesl` is importable as `bevy_elements::noise`; shared libraries are loaded with `load_shader_library!`.
- **GPU timings on a lightly loaded NVIDIA card read 5–10× slow** (P8/P5 clocks under Xvfb). Report the minimum over the diagnostic history and log `nvidia-smi` clocks in the same shell.
- **Screenshots without a display:** `xvfb-run -a -s "-screen 0 1920x1080x24" env ELEMENTS_FRAMES=… ELEMENTS_SHOT=… ./target/release/examples/<name>` renders on the GPU.
- **A `Camera3d` requires `Msaa` (4 samples by default)**, so an observer's `insert_if_new(Msaa::Off)` never applies; `ElementsCamera` inserts `Msaa::Off` unconditionally. The cloud and precipitation passes bind the depth prepass single-sampled and skip a view that has MSAA.
- **Bevy's `ScatteringMedium::earth` falloffs are fractions of the atmosphere thickness** (100 km): effective scale heights 13.3 km (Rayleigh) and 2 km (Mie), not Hillaire's 8 and 1.2 km. Oracles for the sky or cloud light use `bevy_optical_transmittance` (`src/clouds/model.rs`).
- **`ReadbackOnce` lives for one frame**: Bevy removes it in `First`. Spawn it from a system in `Update` (as `gpu/readback.rs` does), never between `app.update()` calls, or use `Readback` and despawn it (`tests/support`). Bevy's readback pool can return a buffer larger than the one asked for: decode only the length you know.
- **An app with no window is not paced**: only presenting a swap chain makes the CPU wait for the GPU, so a headless app (every GPU test) queues frames without bound and readback latency grows until the snapshot freezes. `gpu/readback.rs` `pace_without_window` waits for each windowless frame's GPU work; a stale `Environment` answer in a GPU test means that wait is gone (`docs/research/readback-pacing.md`).
- **`active` is a reserved word in WGSL**: the shader compiler rejects it as an identifier and the kernel then stays idle (look for the `ERROR bevy_elements::gpu::kernel` line).
- **`meta` (and other WGSL reserved words) in a WESL struct field** fails the shader at runtime with "use of a reserved word", not at build time: the terrain material is dark until you read the log for `failed to process shader`.
- **`patch`, `target` and `filter` are WGSL reserved words too**: a local named `patch` in `water.wesl` or a binding named `target` in a kernel fails the shader at runtime (the water or the kernel stays dark/idle until you read the log).
- **`SpecializedMeshPipelineError` has only `MissingVertexAttribute`**, so a material extension cannot reject a view without a depth prepass; the water shader needs `DEPTH_PREPASS` and fails to compile on a view without it.
- **`AlphaMode::Multiply` and `Add` need premultiplied output**: Bevy's `premultiply_alpha` runs in `main_pass_post_lighting_processing`; a shader that writes the result itself (the water's absorb pass) must write `(X·a, a)` for Multiply, because the blend is `dst·src.rgb + (1 − src.a)·dst`.
- **A displaced mesh needs its own `Aabb` and `NoAutoAabb`**, and `MeshTag::new(level)` (the struct has a debug-only private field). The terrain's vertex shader reads the patch origin from the instance matrix's translation, so patches are children of the terrain entity and the terrain `Transform` has no rotation or scale.
- **CDLOD morph needs `m_k ≥ r_{k−1} + diag_k`**, not `0.7 r_k`: a node about to subdivide must be unmorphed over its whole box, else the children pop in and an edge vertex cracks away from its finer neighbour. `terrain::lod::LodConfig::derive` computes the ranges; `adjacent_patches_meet_without_cracks` is the test, and a red plane under the terrain (`terrain_tour` with `ELEMENTS_UNDERLAY=1`) is the visual one.
- **A leaf's min/max must include a one-cell halo**: vertices on its edge interpolate with the neighbour leaf's first cell.
- **Pressure projection with open window edges needs Dirichlet pressure there**: with Neumann on every edge a terrain that blocks one side of the wind window more than the other leaves unprojectable divergence (2.8×10⁻² s⁻¹ over hills, independent of the Jacobi count).
- **`ShaderBuffer::with_size` buffers keep their main-world asset** with `RenderAssetUsages::default()`: replace its `data` to rewrite the GPU copy; `RENDER_WORLD`-only buffers cannot be edited from the main world.
- **Particle and splash buffers** are read back in tests through `PrecipitationState` (`tests/gpu_precipitation.rs`); one test process builds one renderer at a time, so the GPU tests share a lock.
- **Pixel comparisons need `DebandDither::Disabled`**: the tonemapping dither depends on the pixel position, so two cubes with identical materials differ by a level. A render-target `Image` read back with `Readback` needs `TextureUsages::COPY_SRC` added to its descriptor (`Image::new_target_texture` has none), else the render world quits on a validation error (`tests/gpu_weathered.rs`).
- **A material's storage buffers all count against the fragment stage's eight** (Bevy uses five with an atmosphere camera): `WeatheredExtension` binds one, `ElementsGlobals` (80 B, shared, rewritten in place from `update_globals` with `RenderAssetUsages::default()` so no bind group is re-prepared).
- **Layers are written from a test through `Assets<ShaderBuffer>`**: after the terrain is live, `buffer.clear(); buffer.extend_from_slice(&data)` on the `TerrainState` handle re-uploads it with the same buffer identity (same size, label and usage take Bevy's `write_buffer` path). Hold the simulation first (`HydrologySettings { time_scale: 0.0, .. }`), write, then release it; `tests/gpu_hydrology.rs` shows the pattern, and a `time_scale` of `substeps_per_frame · 1.2` runs exactly the quality's substeps in a 1/60 s frame. The GPU tests run at `Ultra` (16 substeps a frame) so a thousand substeps are a hundred frames: the card is shared and a contended frame can take half a second.
- **An f32 water depth cannot hold a transfer below half its ulp**, so rain, evaporation and melt are delivered in quanta of at least 5 µm (`hydrology::model::QuantumAccumulator`), and a deep lake with a surface slope below about 10⁻⁵ does not conserve water exactly (`docs/research/hydrology.md`, Deviation 21). Tests that compare volumes under rain use dry ground or a uniform tank.
- **A pass that dispatches every frame needs a bring-up handshake.** The main world schedules substeps only after the render world says every pipeline is compiled and every buffer exists (`HydrologyGpu::ready`) and counts what was recorded (`executed`) and what was not (`dropped`); wait on `HydrologyStats::substeps`, never on frames.
- **Aeolian transport is a scan, not a field.** `aeolian_flux` solves the flux along each row and column once per step from the upwind edge (one thread per line), because a one-cell-per-sweep Jacobi field cannot both conserve sand and keep the thickness non-negative. The dust passes dispatch hydrology's thermal kernels with their own `HydroParams::avalanche` uniform and run `.after(HydrologyDispatch).before(TerrainDispatch)`. A lone dune does not migrate under the shadow rule (`docs/research/dust.md` Deviation 12); the stripped front of a sand sheet does.

## Build and test

```sh
cargo build --release && cargo test && cargo build --examples --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS=-D warnings cargo doc --no-deps --all-features
cargo build --lib --target wasm32-unknown-unknown
cargo test --release --tests -- --ignored --test-threads=1 --skip ::gpu_tests::   # GPU integration tests, one process per file
for m in clouds terrain cover dust; do cargo test --release --lib --features=isomesh -- --ignored --test-threads=1 $m::gpu_tests::; done   # lib GPU tests, one process per module
ELEMENTS_FRAMES=600 ELEMENTS_SHOT=/tmp/x.png cargo run --release --example quickstart
```

`cargo build --release` runs first in CI because the full `bevy` dev-dependency hides holes in the trimmed feature list.
