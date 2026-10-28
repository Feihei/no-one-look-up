# Local patches to bevy_elements 0.5.3

This directory is a vendored copy of `bevy_elements` 0.5.3, wired in through
`[patch.crates-io]` in the root `Cargo.toml`. Only the cloud shading differs
from upstream; everything else is untouched.

Every change is marked `[PATCH heads-up]` in the source. **Re-apply these on
upgrade** — a straight `cargo update` would silently restore upstream
behaviour and the cloud artefacts would come back.

## Why fork at all

Three cloud defects could not be fixed from the game side. Each was traced to
a specific line before patching:

1. **Every cumulus-like cloud is the same "flat disc with half an egg on top".**
2. **The sky gets brighter as cloud amount rises**, which is backwards.
3. (Related to 1) **Intermediate cloud types do not get intermediate domes.**

### 1 + 3: the dome factor saturated and was a clean half-ellipse

`src/clouds/common.wesl`, `cloud_gradient()`:

```wgsl
let lean = clamp(kind * 4.0, 0.0, 1.0);          // saturates at kind >= 0.25
let scale = mix(1.0, mix(0.3, 1.0, sqrt(dome)), lean);   // perfect ellipse
```

Two problems. `clamp(kind * 4, 0, 1)` reaches full strength at `kind = 0.25`,
and `kind_from_weather` puts most cumulus variants at 0.34–0.50 — so
scattered cumulus, broken cumulus and stratocumulus all got the *identical*
dome, and only a true stratus (`kind` near 0) escaped it. Then `sqrt(dome)`
rises so steeply that the vertical profile is a half-ellipse: the cloud is
full height at the core and falls away in a smooth arc.

Patch:
- the kind ramp is `smoothstep(0.05, 0.55, kind)`, so intermediate types get
  intermediate domes;
- the profile is `pow(dome, 0.62)` (a flatter shoulder) modulated by
  `dome_jitter` in `[0.65, 1.35]`;
- `dome_jitter` comes from the base noise's green channel, which varies per
  blob, so neighbouring clouds are no longer identical;
- `DOME_WIDTH` in `src/clouds/weather.wesl` drops 0.37 -> 0.30, keeping full
  height only in a blob's core.

### 2: `sky_gain` inverted the sky's response to cloud at dusk

`src/celestial/mod.rs`, `sky_gain()` returns `total / beam`, and
`composite.wesl` multiplies sky pixels by it:

```wgsl
cloud.a *= mix(1.0, params.sky.x, smoothstep(OPAQUE_BELOW, SKY_ABOVE, cloud.a));
```

`beam = 1 - cover^3.4`, so the gain climbs steeply with cover — 1.19 at cover
0.78, **1.58 at 0.90**. Upstream's reasoning is that the sky *above* a cloud
deck is not dimmed by the deck below it, which is right at high sun. At dusk
the sun sits ~2.4°, an overcast deck really does stand between the viewer and
the lit sky, and the full gain reads as "more cloud = brighter sky".

Patch: fade the gain out toward the horizon with a smoothstep between 2° and
25° sun altitude. Above 25° nothing changes (upstream's own tests still pass);
at the golden hour the gain is 1.0 and cloud cover stops brightening the sky.

## Verified

- `cargo test` in the game crate: 102 pass.
- Game runs 20 s with zero warnings and zero errors, which also confirms both
  WGSL files compile — a shader error surfaces at pipeline creation.

## Not patched (deliberate)

- **Noise resolution.** The silhouette is quantised at 32–64 m by
  `NOISE_BASE_TEXELS = 128` over an 8192 m tile, `NOISE_DETAIL_TEXELS = 32`
  over 1024 m, and the 512² / 32 km coverage field. Raising these (128³ ->
  256³) would sharpen cloud edges, at roughly +128 MB of VRAM. Left alone
  until the shape changes above are judged by eye; edges and shape are
  separate problems and mixing them makes the result hard to attribute.
- **Continuous decks going black at dusk.** At 2.4° sun altitude, light has to
  cross the whole layer at a grazing angle; the fix on the game side is to
  keep sheet-like kinds thin (see `CloudKind::layer`), not to fake ambient
  into the shader.
