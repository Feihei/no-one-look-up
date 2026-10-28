# Changelog

All notable changes to bevy_elements. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate is pre-1.0, so any release may break the API.

## [Unreleased]

## [0.5.3] — 2026-10-16

### Added

- **The weather can be saved and restored.** `WeatherDirector` derives `Reflect`, `PartialEq` and, with the `serde` feature, `Serialize`/`Deserialize`. Saved with `WorldClock`, `Climate` and `GeoLocation` and inserted into a fresh app, it continues the same scheduler chain, forced preset and blend in progress, where a fresh director restarted at `Clear` and walked a different chain. `docs/research/weather-save.md`, `tests/save.rs`.

## [0.5.2] — 2026-10-16

### Fixed

- **Gameplay queries stay fresh in an app with no window.** Nothing paced an app that renders only to images (a headless server, a simulation tool, every GPU test): its frames were submitted faster than the GPU retired them, the latency of each readback doubled every few requests, and in the ground-cover suite no readback arrived again after three seconds, so `wind_at`, `water_depth`, `snow_depth`, `wetness`, `cloud_shadow_at` and the map-backed `SurfaceWeathering` answered with the state of the first seconds. A frame with no window now waits for its own GPU work, so every readback lands on the next frame; windowed apps, which presenting already paces, are untouched. This also fixes the three GPU tests that failed on `main` since the loop's iteration 4 (`cover::gpu_wetness_soaks_and_dries_on_the_formula`, `cover::gpu_surface_weathering_follows_the_map`, `voxel::gpu_voxel_snow_covers_the_top`). `docs/research/readback-pacing.md`.

## [0.5.1] — 2026-10-16

### Added

- **`Environment::cloud_shadow_at(Vec3)` and `Environment::illuminance_at(Vec3)`**: the cloud's shadow and the horizontal illuminance (lx) at a world position, for stealth, shade-seeking, solar or growth gameplay. The cloud shadow kernel writes every fourth texel of the shadow map into a 256² grid (31.25 m cells over the map's 8 km) that the 4 Hz readback copies; the query projects the position along the light as the ground's material does, and is 1 outside the map. The illuminance is the clear-sky sun and moon light times Kasten & Czeplak's `0.25 + 0.75 · T` at that point. Clouds only: terrain and meshes do not shadow either answer. `docs/research/cloud-shadow-query.md`.
- **`cloud_shadow_probes` example**: a grid of unlit markers coloured by `cloud_shadow_at` (yellow open, blue shadowed) over the cloud shadows on the ground, with the illuminance at the grid's centre on screen.

### Fixed

- **`day_night` now frames the sun and the moon.** The example's camera was aimed west-south-west
  at a fixed bearing, and the vision review measured every body outside that frustum for the
  whole sweep — sunrise at azimuth 54–65° behind the observer, noon at 180°/68° up, sunset at
  304° past the right edge, the waxing moon at 150–175° south: the domain that exists to render
  those discs never showed one. The camera now aims at the sun while it is above the horizon,
  at the moon once the sun is down and the moon is up, and back at the evening heading when
  neither is up; a paused hour (`ELEMENTS_HOUR`) aims once so the still is framed, and holding
  the right button releases the aim.
- **Snow relief reads under an overcast sky.** Saturated snow rendered as a flat wash (0.56 DN
  of standard deviation across 840 k pixels of terrain from the air) because the ambient is
  uniform over normals and the beam is ~0 under cloud. The crate's materials now shade the lit
  colour by the Moon & Spencer (1942) CIE overcast dome — brightest at the zenith, the horizon a
  third of it, integrated with the ground's bounce into the anchors top 1 / side 0.68 / bottom
  0.5 — ramped by the cloud map's transmittance over each ground point, so clear air is an exact
  no-op (controls within dither) and a full deck gets the whole shape. The showcase's boxes and
  columns join its ground on `WeatheredMaterial`, so the shading reaches them: their sides read
  9 DN darker than their tops under Snow while the tops do not move. Under Overcast 559 652
  terrain pixels move ≥ 3 DN (13 726 ≥ 7) where none did before (`docs/research/snow-cover.md`,
  Deviation 15).

- **A dust storm streams.** The dust sprites were drawn at their 2 mm grain size — 0.02–0.1 px
  anywhere in the 60 m box — so the sub-pixel coverage rule faded them out and the storm read as
  static fog (zero pixels below 150 against a 224 background across a 1150 × 140 sky band).
  Sprites are now drawn at a 0.35 m cluster scale with their own 0.5 opacity (below a flake's
  0.8, so a cluster sprite is a veil, not a puff); the grain size, the sprite extinction, the
  counts and the motion are unchanged (`docs/research/dust.md`, Deviation 15).
- **The horizon shows no dark band where the ground ends before it.** Past the edge of a finite ground (quickstart hills from 6 m eye, any terrain that does not reach the horizon) the below-horizon sky drew Bevy's unlit planet term: `ground_haze` fogged over the bare planet's near root — 23 m at 5° below the horizon from a 2 m eye, a transmittance of ~1 — so a 45-row band read 157–171 against sky 243 and fogged ground 229 under `Snow` and `Rain`. The fog distance is now floored at the march range (32 km): a below-horizon pixel with no mesh shows ground the fog has fully carried, because nothing nearer than the far field is visible beyond a finite terrain. New GPU test `the_horizon_sky_is_the_fog_colour_under_a_deck` pins both sides of the horizon (`docs/research/clouds.md`, Deviation 28).
- **Snow now lies on the ground the examples show, and a still can show it.** `weather_tour` under `Snow` and `Blizzard` rendered olive-drab ground where the boxes and columns beside it were white: the harness's 2 km plane was a bespoke material that sampled only the cloud shadow map. The examples' ground is now `WeatheredMaterial` — it darkens and glosses in rain, runs with streaks and whitens under snow as a consumer's mesh does — and its plane reaches 40 km, so the ground meets the sky without the sliver of unlit below-horizon atmosphere a 2 km plane leaves at the horizon (a 66 DN drop to a band floor 16 DN below the ground under `Overcast` and `PartlyCloudy`, now gone). A held clock (`ELEMENTS_HOUR`) integrates no game time, so no material could ever whiten in a still; `ELEMENTS_GROW_HOURS=n` now runs `n` game hours of the held preset before the first frame and rewinds the clock, leaving the hour, sun and sky as the example's own (`docs/research/snow-cover.md`, Deviation 14; `docs/research/clouds.md`, Deviation 28).
- **The sky carries the weather's air, so a whiteout has no horizon line.** The distance fog fogs every mesh to the fog colour as it recedes, but the sky kept the cloud deck's own radiance: under `Blizzard` the horizon read as a 40-level step (sky `(186, 193, 198)` above fogged ground `(224, 228, 230)`), and the same step showed on real terrain and from the air. The composite now fogs sky pixels by `σ_air = σ_fog − 3.912/40 km`, read from the camera's own `DistanceFog` and with the clear-air floor removed, so the sky converges on the same fog colour the ground does, and `Clear` (40 km) is a bit-exact no-op. `fog::sky_air_extinction` is the one authority for the floor (`docs/research/clouds.md`, Deviation 29).
- **The ground-fog volume no longer darkens the sky it cannot light.** Bevy lights a `FogVolume` from the `DirectionalLight`s alone, which under cloud carry only the beam `T̄` — zero at full cover. The volume therefore removed the sky's light and returned none of it, so a short `Weather::visibility_m` turned the whole sky dark: a noon blizzard rendered as a dark hole (measured zenith `(46, 47, 50)`). Its density now fades by the beam's share of the scene's light and its intensity rises by the reciprocal, so it keeps its radiance-to-opacity ratio and simply fades out where Bevy cannot light it; the sky and the distance fog draw the whiteout (`docs/research/fog.md`, Deviation 7). `Clear` and thin-cloud presets are unchanged.

## [0.5.0] — 2026-10-14

Everything from 0.2 through 0.5 lands in one release. Every system was studied in a committed research note first (`docs/research/`, one per system: the papers, the equations adopted, every deviation with its source, and the numbers the tests assert).

### Added

- **Heightfield terrain (`TerrainPlugin`).** A `Terrain` from procedural presets (`hills`, `mountains`, `dunes`, `plains`), a height map or level ground, generated on the GPU into per-cell layers and drawn with a CDLOD patch pool (Strugar 2009) and a slope/sediment layer-blending material; `TerrainRender::External` still simulates when your own mesh holds the surface, shaded through `GroundExtension`. `Environment::terrain_height` reads the height back.
- **Hydrology and erosion (`HydrologyPlugin`).** Flooding, rivers and lakes as one shallow-water field on the terrain grid (Mei et al. 2007 virtual pipes with a depth-dependent pipe area and Manning friction), `WaterSource` and `WaterSink`, rain split into rain and snow per cell, hydraulic erosion with flux-form sediment transport, thermal erosion that leaves rock faces standing on game time, and `ErodeTerrain` bakes of simulated time. `Environment::water_depth` and `water_velocity`.
- **Water rendering (`WaterPlugin`).** The water drawn over the terrain by a second CDLOD patch pool: a per-channel Beer–Lambert *absorb* pass with thickness from the depth prepass corrected for refraction, and a *surface* pass with the Fresnel reflection of the sky, the silt body colour, foam and flow-mapped ripples (Vlachos 2010) from the simulated velocity; the sun's path through the water shades the bed.
- **Snow and wetness maps (`GroundCoverPlugin`).** Snow as a depth and water equivalent per cell, stepped on game time: each cell's own rain–snow split, slope shed and blocker shelter, densification, wind drift, smoothing, and degree-day melt that arrives in hydrology's water exactly (Hock 2003). Wetness from rain and evaporation. The terrain and ground materials shade snow, wet darkening and gloss, and puddles with ripples; `SurfaceWeathering` (and so `WeatheredMaterial`) follows the maps' mean around the camera. `Environment::snow_depth` and `wetness`.
- **Weathered materials (`WeatheredPlugin`).** `WeatheredMaterial` for any mesh: wet darkening and gloss, rain streaks, up-facing snow on flora's curve, and the cloud shadows, fed by one shared `ElementsGlobals` buffer with a per-material `exposure` for shelter.
- **Dust storms (`DustPlugin`).** Aeolian transport on the terrain's sediment layer: Paris et al.'s capacity with Venturi speed-up and a lee shadow, a flux line scan that conserves sand exactly, and hydrology's thermal kernels avalanching at the sand's repose angle; dust sprites in the precipitation buffer and a fog tint while `Weather::dust` is up.
- **Snow on voxel volumes (feature `isomesh`, `VoxelPlugin`).** `VoxelGround` on a `bevy_isomesh` `VoxelVolume` scans the column tops into a crate-internal `Terrain` (`TerrainRender::External`) the snow, wetness and hydrology run on, and a `SnowyField` adds the snow depth back as a slab over the tops, swapped per column when it moves half a voxel and remeshing only the chunks that changed; tunnels stay clear. Erosion is off for voxel ground. Needs `isomesh`/`bevy_isomesh` 0.0.11.

### Changed

- Overcast light: the sun and moon `DirectionalLight`s carry the cloud's beam transmittance and the diffuse skylight remainder is the camera `AmbientLight`, so object shadows soften under cloud; the sky reads grey, not blue.
- The far field of the clouds is lit and seen through the air exactly as the marched deck is at the fade boundary, and a ray that meets the planet past the edge of a finite ground draws the camera's distance fog instead of a cloud.
- Terrain layer blend: Andersson-style Overlay detail on wider ramps, so rock, soil and grass meet in irregular mottled transitions instead of iso-slope blobs.
- Snowflakes are soft discs of bounded elongation, not dashes; snowflakes' streak is capped at one diameter.
- `profile.dev` sets `debug = 0`: Bevy examples' debuginfo links OOM CI runners.

### Examples

New: `terrain_tour`, `external_ground`, `flood`, `lake`, `erosion_bake`, `weathered_town`, `ground_cover`, `dust_storm`, `voxel_snow` (feature `isomesh`), and `quickstart` now stands on hills. `bench_elements` times every pass, with `ELEMENTS_TERRAIN=hills|mountains|dunes|lake|none`.

[0.5.1]: https://github.com/Ladvien/bevy_elements/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Ladvien/bevy_elements/compare/v0.1.0...v0.5.0