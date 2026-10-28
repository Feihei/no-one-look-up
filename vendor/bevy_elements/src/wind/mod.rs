//! GPU wind: a camera-centred stable-fluids field relaxed toward the weather's wind, with curl
//! noise turbulence and travelling gusts. Shaders sample it through the public
//! `bevy_elements::wind` WGSL import; gameplay reads it through
//! [`Environment::wind_at`](crate::Environment::wind_at).

mod gpu;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::asset::embedded_asset;
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use bevy::render::storage::ShaderBuffer;

use crate::climate::{Climate, Weather};
use crate::gpu::grid::{storage_buffer, storage_volume};
use crate::gpu::readback::{Readbacks, decode};
use crate::plugin::{ElementsQuality, ElementsSystems, ElementsView, ensure_gpu_core};
use crate::query::EnvironmentSnapshot;
use crate::terrain::{NO_TERRAIN, TerrainParamsGpu, TerrainState};

/// Wind cell edge, m (horizontal and vertical).
pub const WIND_CELL_M: f32 = 4.0;
/// Roughness length of the forcing profile, m (WMO open terrain).
pub const WIND_ROUGHNESS_M: f32 = 0.03;
/// Most gust bands active at once.
pub const MAX_GUSTS: usize = 16;
/// Forcing relaxation rate, 1/s.
const RELAX_RATE: f32 = 0.5;
/// Curl-noise length scale of the coarse octave, m (the fine octave is half).
const CURL_SCALE_M: f32 = 32.0;
/// Scales `curl3` of the shared noise to unit rms; measured over 10⁵ points by
/// `the_kernel_normalisation_constants_match_the_noise`.
pub(crate) const CURL_NORMALISE: f32 = 0.709;
/// Curl-noise amplitude per unit gustiness, as a fraction of u10.
const CURL_FRACTION: f32 = 0.35;
/// Gusts per second per unit gustiness.
const GUST_RATE: f32 = 0.1;
/// Gust bands start this far upwind of the camera, m.
const GUST_SPAWN_M: f32 = 200.0;
/// Gust band speed as a multiple of u10.
const GUST_SPEED: f32 = 1.2;
/// Gust band width, m.
const GUST_WIDTH_M: f32 = 30.0;
/// Gust peak as a fraction of gustiness × u10.
const GUST_STRENGTH: f32 = 0.5;
/// The shared noise lattice is periodic with this many cells, so the drifting offset wraps
/// seamlessly and stays small enough for f32.
const NOISE_WRAP: f32 = 4096.0;

/// Mean wind speed at `height_m` above ground for `u10` at 10 m, m/s: the neutral log law
/// `u10 · ln(max(z, z0)/z0) / ln(10/z0)` with z0 = 0.03 m.
pub fn log_wind_speed(u10: f32, height_m: f32) -> f32 {
    let z0 = WIND_ROUGHNESS_M;
    u10 * (height_m.max(z0) / z0).ln() / (10.0 / z0).ln()
}

/// The uniform the `bevy_elements::wind` WGSL import takes. Bind it with
/// `#[uniform(N)] wind: WindFieldParams` and fill it from [`WindState::field_params`].
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType, Reflect)]
pub struct WindFieldParams {
    /// xyz: world position of the field's minimum corner, m. w: unused.
    pub origin: Vec4,
    /// xyz: field size, m. w: 1 when the field holds simulated data, else 0.
    pub extent: Vec4,
    /// xyz: prevailing wind at 10 m, m/s, used outside the field. w: unused.
    pub prevailing: Vec4,
}

/// The wind, for shaders, sibling crates and gameplay.
#[derive(Resource, Clone, Debug, Reflect)]
pub struct WindState {
    /// Unit XZ direction the prevailing wind blows toward.
    pub prevailing: Vec2,
    /// Mean wind speed at 10 m, m/s.
    pub speed_10m: f32,
    /// Peak 3-s gust over the mean: `1 + 0.5·gustiness`.
    pub gust_factor: f32,
    /// Wind at the followed camera, m/s.
    pub at_camera: Vec3,
    /// The simulated field: a 3D `rgba16float` texture, xyz = velocity m/s, w = turbulence
    /// intensity, toroidally addressed (see `bevy_elements::wind`).
    pub field: Handle<Image>,
    /// World position of the field's minimum corner, m.
    pub field_origin: Vec3,
    /// Field size, m.
    pub field_extent: Vec3,
    /// Whether the field has been simulated at least once.
    pub field_live: bool,
}

impl Default for WindState {
    fn default() -> Self {
        Self {
            prevailing: Vec2::X,
            speed_10m: 0.0,
            gust_factor: 1.0,
            at_camera: Vec3::ZERO,
            field: Handle::default(),
            field_origin: Vec3::ZERO,
            field_extent: Vec3::ZERO,
            field_live: false,
        }
    }
}

impl WindState {
    /// The uniform for the `bevy_elements::wind` WGSL import.
    pub fn field_params(&self) -> WindFieldParams {
        WindFieldParams {
            origin: self.field_origin.extend(0.0),
            extent: self
                .field_extent
                .extend(if self.field_live { 1.0 } else { 0.0 }),
            prevailing: self.prevailing_velocity().extend(0.0),
        }
    }

    /// The prevailing wind at 10 m as a world velocity, m/s.
    pub fn prevailing_velocity(&self) -> Vec3 {
        Vec3::new(self.prevailing.x, 0.0, self.prevailing.y) * self.speed_10m
    }
}

/// A gust band started: it travels from `origin` along `direction`.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WindGust {
    /// World XZ where the band's centre line starts, m.
    pub origin: Vec2,
    /// Unit XZ direction the band travels and pushes.
    pub direction: Vec2,
    /// Travel speed, m/s.
    pub speed: f32,
    /// Band width across its travel direction, m.
    pub width: f32,
    /// Peak added wind speed at 10 m, m/s.
    pub strength: f32,
}

/// A CPU copy of the wind field at one readback.
#[derive(Clone, Debug)]
pub(crate) struct WindSnapshot {
    origin_cell: IVec3,
    size: UVec3,
    data: Arc<[Vec4]>,
}

impl WindSnapshot {
    fn slot(&self, w: IVec3) -> usize {
        let n = self.size.as_ivec3();
        let s = w.rem_euclid(n);
        (s.x + n.x * (s.y + n.y * s.z)) as usize
    }

    /// Trilinear velocity at a world position, or `None` outside the window.
    pub fn sample(&self, p: Vec3) -> Option<Vec3> {
        let g = p / WIND_CELL_M - self.origin_cell.as_vec3() - 0.5;
        let hi = self.size.as_vec3() - 1.0;
        if g.cmplt(Vec3::splat(-0.5)).any() || g.cmpgt(hi + 0.5).any() {
            return None;
        }
        let g = g.clamp(Vec3::ZERO, hi);
        let b = g.floor().as_ivec3();
        let f = g - g.floor();
        let max = self.size.as_ivec3() - 1;
        let at = |o: IVec3| -> Vec3 {
            let c = (b + o).min(max);
            self.data
                .get(self.slot(self.origin_cell + c))
                .copied()
                .unwrap_or(Vec4::ZERO)
                .truncate()
        };
        let x00 = at(IVec3::new(0, 0, 0)).lerp(at(IVec3::new(1, 0, 0)), f.x);
        let x10 = at(IVec3::new(0, 1, 0)).lerp(at(IVec3::new(1, 1, 0)), f.x);
        let x01 = at(IVec3::new(0, 0, 1)).lerp(at(IVec3::new(1, 0, 1)), f.x);
        let x11 = at(IVec3::new(0, 1, 1)).lerp(at(IVec3::new(1, 1, 1)), f.x);
        Some(x00.lerp(x10, f.y).lerp(x01.lerp(x11, f.y), f.z))
    }
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub(crate) struct GustGpu {
    a: Vec4,
    b: Vec4,
}

/// Mirrors `Params` in `sim.wesl`.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub(crate) struct WindParams {
    size: UVec4,
    pub(super) origin: IVec4,
    prev_origin: IVec4,
    wind: Vec4,
    forcing: Vec4,
    misc: Vec4,
    noise_offset: Vec4,
    terrain: TerrainParamsGpu,
    gusts: [GustGpu; MAX_GUSTS],
}

/// The GPU resources of the wind field and this frame's parameters; extracted to the render
/// world every frame.
#[derive(Resource, Clone)]
pub(crate) struct WindGpu {
    pub size: UVec3,
    pub velocity: [Handle<ShaderBuffer>; 2],
    pub pressure: [Handle<ShaderBuffer>; 2],
    pub solid: Handle<ShaderBuffer>,
    pub divergence: Handle<ShaderBuffer>,
    pub field: Handle<Image>,
    /// The terrain's `terrain_cells`, or the one-cell placeholder without a live terrain.
    pub terrain: Handle<ShaderBuffer>,
    pub params: WindParams,
    pub iterations: u32,
    pub active: bool,
    /// Set by the render world after the kernels first ran; until then the buffers hold no
    /// simulated wind and are neither read back nor published as live.
    pub live: Arc<AtomicBool>,
}

#[derive(Resource, Default)]
struct WindRuntime {
    origin: Option<IVec3>,
    reset: bool,
    noise_offset: Vec3,
    gusts: Vec<(WindGust, f32)>,
    rng: u64,
}

/// The GPU wind field.
///
/// Needs an [`ElementsCamera`](crate::ElementsCamera) to centre on; without one the field idles
/// and [`WindState`] reports the prevailing wind.
pub struct WindPlugin;

impl Plugin for WindPlugin {
    fn build(&self, app: &mut App) {
        ensure_gpu_core(app);
        embedded_asset!(app, "sim.wesl");
        app.init_resource::<WindRuntime>()
            .add_message::<WindGust>()
            .add_systems(
                Update,
                (ensure_grid, step_wind)
                    .chain()
                    .in_set(ElementsSystems::Simulation),
            )
            .add_systems(Update, read_wind.in_set(ElementsSystems::Readback));
        gpu::plugin(app);
    }
}

fn ensure_grid(
    mut commands: Commands,
    quality: Res<ElementsQuality>,
    grid: Option<Res<WindGpu>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut images: ResMut<Assets<Image>>,
    mut runtime: ResMut<WindRuntime>,
    mut state: ResMut<WindState>,
) {
    let size = quality.wind_grid();
    if grid.is_some_and(|g| g.size == size) {
        return;
    }
    let n = (size.x * size.y * size.z) as usize;
    let field = storage_volume(&mut images, size);
    state.field = field.clone();
    state.field_live = false;
    runtime.reset = true;
    commands.insert_resource(WindGpu {
        size,
        velocity: [
            storage_buffer(&mut buffers, "elements_wind_velocity_a", vec![[0f32; 4]; n]),
            storage_buffer(&mut buffers, "elements_wind_velocity_b", vec![[0f32; 4]; n]),
        ],
        pressure: [
            storage_buffer(&mut buffers, "elements_wind_pressure_a", vec![0f32; n]),
            storage_buffer(&mut buffers, "elements_wind_pressure_b", vec![0f32; n]),
        ],
        solid: storage_buffer(&mut buffers, "elements_wind_solid", vec![0u32; n]),
        divergence: storage_buffer(&mut buffers, "elements_wind_divergence", vec![0f32; n]),
        field,
        terrain: NO_TERRAIN,
        params: WindParams::default(),
        iterations: quality.wind_pressure_iterations(),
        active: false,
        live: Arc::new(AtomicBool::new(false)),
    });
}

fn next_unit(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (crate::climate::mix64(*state) >> 40) as f32 / 16_777_216.0
}

/// Gusts: Poisson starts at `GUST_RATE · gustiness` per second, upwind of the camera; bands
/// advance and expire after crossing twice the spawn distance. Returns the gust started this
/// step, if any.
fn advance_gusts(
    runtime: &mut WindRuntime,
    camera: Vec3,
    direction: Vec2,
    u10: f32,
    gustiness: f32,
    dt: f32,
) -> Option<WindGust> {
    let p_start = 1.0 - (-GUST_RATE * gustiness * dt).exp();
    let started = (next_unit(&mut runtime.rng) < p_start).then(|| WindGust {
        origin: camera.xz() - direction * GUST_SPAWN_M,
        direction,
        speed: GUST_SPEED * u10,
        width: GUST_WIDTH_M,
        strength: GUST_STRENGTH * gustiness * u10,
    });
    if let Some(gust) = started {
        runtime.gusts.push((gust, 0.0));
        if runtime.gusts.len() > MAX_GUSTS {
            runtime.gusts.remove(0);
        }
    }
    for (_, travelled) in &mut runtime.gusts {
        *travelled += dt;
    }
    runtime
        .gusts
        .retain(|(g, t)| g.speed * *t < 2.0 * GUST_SPAWN_M + g.width);
    started
}

#[allow(clippy::too_many_arguments)]
fn step_wind(
    time: Res<Time<Virtual>>,
    view: Res<ElementsView>,
    weather: Res<Weather>,
    climate: Res<Climate>,
    snapshot: Res<EnvironmentSnapshot>,
    terrain: Option<Res<TerrainState>>,
    mut grid: Option<ResMut<WindGpu>>,
    mut runtime: ResMut<WindRuntime>,
    mut state: ResMut<WindState>,
    mut gusts: MessageWriter<WindGust>,
) {
    let u10 = weather.wind_speed_ms;
    let direction = weather.wind_direction.normalize_or(Vec2::X);
    let gustiness = weather.gustiness.clamp(0.0, 1.0);
    let dt = time.delta_secs();
    state.prevailing = direction;
    state.speed_10m = u10;
    state.gust_factor = 1.0 + GUST_STRENGTH * gustiness;

    let Some(grid) = grid.as_deref_mut() else {
        return;
    };
    let Some(camera) = view.entity.map(|_| view.position) else {
        grid.active = false;
        state.at_camera = state.prevailing_velocity();
        return;
    };

    if runtime.rng == 0 {
        runtime.rng = climate.seed ^ 0x57_1D_6A_57;
    }
    if let Some(gust) = advance_gusts(&mut runtime, camera, direction, u10, gustiness, dt) {
        gusts.write(gust);
    }

    // Window: whole-cell steps, centred horizontally. Vertically it starts 8 m below the ground
    // under the camera, or lower as the camera rises, so it never sits more than three quarters
    // of its height under the camera (`terrain.md`: the ground under the camera is the snapshot
    // surface, the camera 8 m up before the first readback, the plane y = 0 without a terrain).
    let size = grid.size.as_ivec3();
    let ground = match (&terrain, snapshot.terrain.as_ref()) {
        (Some(_), Some(t)) => t.height(camera.xz()).unwrap_or(camera.y - 8.0),
        (Some(_), None) => camera.y - 8.0,
        (None, _) => 0.0,
    };
    let window_floor = (ground - 8.0).max(camera.y - 0.75 * grid.size.y as f32 * WIND_CELL_M);
    let cam_cell = (camera / WIND_CELL_M).floor().as_ivec3();
    let origin = IVec3::new(
        cam_cell.x - size.x / 2,
        (window_floor / WIND_CELL_M).floor() as i32,
        cam_cell.z - size.z / 2,
    );
    let prev = runtime.origin.unwrap_or(origin);
    let reset = std::mem::take(&mut runtime.reset) || runtime.origin.is_none();
    runtime.origin = Some(origin);

    let drift = Vec3::new(direction.x, 0.0, direction.y) * u10 * dt / CURL_SCALE_M
        + Vec3::new(0.0, 0.05 * dt, 0.0);
    runtime.noise_offset = (runtime.noise_offset - drift).rem_euclid(Vec3::splat(NOISE_WRAP));

    let mut bands = [GustGpu::default(); MAX_GUSTS];
    for (slot, (g, t)) in bands.iter_mut().zip(&runtime.gusts) {
        let centre = g.origin + g.direction * g.speed * *t;
        *slot = GustGpu {
            a: Vec4::new(centre.x, centre.y, g.direction.x, g.direction.y),
            b: Vec4::new(g.width, g.strength, 0.0, 0.0),
        };
    }
    let live = terrain.as_ref().filter(|t| t.is_live());
    let terrain_params = live.map_or_else(TerrainParamsGpu::default, |t| {
        TerrainParamsGpu::new(&t.grid, 1, true)
    });
    grid.terrain = live.map_or(NO_TERRAIN, |t| t.cells.clone());
    grid.params = WindParams {
        size: grid.size.extend(0),
        origin: origin.extend(i32::from(reset)),
        prev_origin: prev.extend(0),
        wind: Vec4::new(direction.x, direction.y, u10, gustiness),
        forcing: Vec4::new(
            WIND_ROUGHNESS_M,
            1.0 - (-RELAX_RATE * dt).exp(),
            CURL_FRACTION * gustiness * u10 * CURL_NORMALISE,
            CURL_SCALE_M,
        ),
        misc: Vec4::new(WIND_CELL_M, dt, 0.0, runtime.gusts.len() as f32),
        noise_offset: runtime.noise_offset.extend(0.0),
        terrain: terrain_params,
        gusts: bands,
    };
    grid.active = dt > 0.0;
    state.field_origin = origin.as_vec3() * WIND_CELL_M;
    state.field_extent = grid.size.as_vec3() * WIND_CELL_M;
    state.field_live = grid.live.load(Ordering::Relaxed);
    state.at_camera = snapshot
        .wind
        .as_ref()
        .and_then(|w| w.sample(camera))
        .unwrap_or_else(|| {
            Vec3::new(direction.x, 0.0, direction.y) * log_wind_speed(u10, camera.y)
        });
}

fn read_wind(grid: Option<Res<WindGpu>>, mut readbacks: Readbacks) {
    let Some(grid) = grid else {
        return;
    };
    if !grid.active || !grid.live.load(Ordering::Relaxed) {
        return;
    }
    let origin_cell = grid.params.origin.truncate();
    let size = grid.size;
    readbacks.request("wind", grid.velocity[0].clone(), move |bytes, snapshot| {
        let Some(cells) = decode::<[f32; 4]>(bytes) else {
            return;
        };
        if cells.len() != (size.x * size.y * size.z) as usize {
            return;
        }
        snapshot.wind = Some(WindSnapshot {
            origin_cell,
            size,
            data: cells.into_iter().map(Vec4::from_array).collect(),
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_log_law_matches_the_research_note() {
        let u = |z| log_wind_speed(1.0, z);
        assert!((u(10.0) - 1.0).abs() < 1e-6);
        assert!((u(1.0) - 0.6036).abs() < 1e-3);
        assert!((u(2.0) - 0.7229).abs() < 1e-3);
        assert!((u(5.0) - 0.8807).abs() < 1e-3);
        assert!((u(20.0) - 1.1193).abs() < 1e-3);
        assert!((u(50.0) - 1.2771).abs() < 1e-3);
        assert_eq!(u(0.01), 0.0);
    }

    #[test]
    fn the_snapshot_is_toroidal_and_bounded() {
        let size = UVec3::new(4, 2, 4);
        let origin_cell = IVec3::new(-2, 0, 7);
        let n = (size.x * size.y * size.z) as usize;
        let mut data = vec![Vec4::ZERO; n];
        let snap = WindSnapshot {
            origin_cell,
            size,
            data: Arc::from(vec![Vec4::ZERO; n]),
        };
        // World cell (−1, 1, 9) → slot ((−1 mod 4), 1, (9 mod 4)) = (3, 1, 1).
        let s = snap.slot(IVec3::new(-1, 1, 9));
        // x + nx * (y + ny * z) with (x, y, z) = (3, 1, 1), nx = 4, ny = 2.
        assert_eq!(s, 3 + 4 * (1 + 2));
        data[s] = Vec4::new(5.0, 0.0, 0.0, 0.0);
        let snap = WindSnapshot {
            data: Arc::from(data),
            ..snap
        };
        let centre = (IVec3::new(-1, 1, 9).as_vec3() + 0.5) * WIND_CELL_M;
        assert_eq!(snap.sample(centre), Some(Vec3::new(5.0, 0.0, 0.0)));
        assert_eq!(snap.sample(Vec3::new(1000.0, 0.0, 0.0)), None);
    }

    /// Validation 5 of `docs/research/wind.md`: with gustiness 0.9 and u10 = 10 m/s a band is
    /// 4.5 m/s strong, travels at 12 m/s, is 30 m wide and starts 200 m upwind of the camera;
    /// the rate is 0.09 s⁻¹, so an hour holds 324 ± 72 (4σ) starts, and no band outlives its
    /// crossing of (2 · 200 + 30) m.
    #[test]
    fn gusts_start_upwind_at_the_poisson_rate_and_expire_after_crossing() {
        let mut runtime = WindRuntime {
            rng: 0x1234_5678,
            ..WindRuntime::default()
        };
        let camera = Vec3::new(100.0, 5.0, -40.0);
        let direction = Vec2::new(3.0, 4.0).normalize();
        let dt = 1.0 / 60.0;
        let lifetime = (2.0 * GUST_SPAWN_M + GUST_WIDTH_M) / 12.0;
        let mut started = 0;
        for _ in 0..3600 * 60 {
            if let Some(g) = advance_gusts(&mut runtime, camera, direction, 10.0, 0.9, dt) {
                started += 1;
                assert!((g.strength - 4.5).abs() < 1e-5);
                assert!((g.speed - 12.0).abs() < 1e-5);
                assert_eq!(g.width, 30.0);
                assert!(
                    g.origin
                        .distance(Vec2::new(100.0, -40.0) - direction * 200.0)
                        < 1e-4
                );
            }
            assert!(runtime.gusts.iter().all(|(_, t)| *t <= lifetime + dt));
        }
        assert!((250..=400).contains(&started), "{started} gusts in an hour");
    }
}
