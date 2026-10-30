//! The strike process: a seeded inhomogeneous Poisson process on `Time<Virtual>`, scripted
//! strikes, the messages and the delayed thunder. CPU only; runs without a renderer.

use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use super::bolt::{self, Strokes};
use crate::climate::{Climate, Weather, hash64};
use crate::clouds::CloudLayer;
use crate::plugin::{ElementsUnits, ElementsView};
use crate::query::EnvironmentSnapshot;

/// Decouples the strike streams from the hourly weather scheduler, which hashes the same way:
/// the ASCII bytes of "lightnin".
pub(crate) const DOMAIN: u64 = 0x6C69_6768_746E_696E;
/// Radius of the disc strikes fall in, centred on the followed camera, m.
pub const STRIKE_RADIUS_M: f32 = 8000.0;
/// Speed of sound used for the thunder delay, m/s.
pub const SPEED_OF_SOUND_MS: f32 = 343.0;

/// Script a strike: lightning hits `at` (the attachment point, in world units; its `y` is the
/// ground height) on the next frame.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StrikeLightning {
    /// Where the bolt attaches, world units.
    pub at: Vec3,
}

/// A bolt struck, at the moment of the flash.
#[derive(Message, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LightningStrike {
    /// The attachment point, world units.
    pub position: Vec3,
    /// The bolt's shape seed: the same seed and attachment point give the same bolt.
    pub seed: u64,
}

/// Thunder reaches the followed camera: written `distance_m / 343` s of real time after the
/// matching [`LightningStrike`].
#[derive(Message, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Thunder {
    /// Where the strike hit, world units.
    pub position: Vec3,
    /// Distance from the followed camera to the strike when it struck, m.
    pub distance_m: f32,
    /// Loudness, `1 / (1 + distance_m / 1000)`: 0.5 at 1 km, 0.11 at 8 km.
    pub loudness: f32,
}

/// `u(s, i, k)`: the open-interval uniform for stream `k` of strike `i`.
fn unit(seed: u64, index: u64, k: u64) -> f64 {
    ((hash64(seed ^ DOMAIN, index * 8 + k) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// One strike drawn by the process.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Draw {
    /// Virtual time of the strike, s.
    pub time: f64,
    /// Offset from the followed camera in XZ, m.
    pub offset: Vec2,
    /// Bolt shape seed.
    pub seed: u64,
}

/// The cumulative-hazard construction of the inhomogeneous Poisson process: strike `i` happens
/// when `∫ λ dt` has grown by the `i`-th unit exponential since the previous strike.
#[derive(Clone, Debug)]
pub(crate) struct StrikeProcess {
    seed: u64,
    index: u64,
    /// Hazard still to accumulate before the next strike.
    need: f64,
}

impl StrikeProcess {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            seed,
            index: 0,
            need: -unit(seed, 0, 0).ln(),
        }
    }

    /// Advance over the frame `[start, start + dt]` at a constant `rate` (s⁻¹) and push every
    /// strike in it, its time interpolated within the frame.
    pub(crate) fn advance(&mut self, start: f64, dt: f64, rate: f64, out: &mut Vec<Draw>) {
        let mut left = rate * dt;
        let mut t = start;
        while left >= self.need {
            t += self.need / rate;
            left -= self.need;
            let i = self.index;
            let r = f64::from(STRIKE_RADIUS_M) * unit(self.seed, i, 1).sqrt();
            let bearing = std::f64::consts::TAU * unit(self.seed, i, 2);
            out.push(Draw {
                time: t,
                offset: Vec2::new((r * bearing.cos()) as f32, (r * bearing.sin()) as f32),
                seed: hash64(self.seed ^ DOMAIN, i * 8 + 3),
            });
            self.index += 1;
            self.need = -unit(self.seed, self.index, 0).ln();
        }
        self.need -= left;
    }
}

/// A flash in progress: its shape, its strokes and where the clock is. Despawned when the
/// envelope has died out.
#[derive(Component, Debug)]
pub(crate) struct ActiveBolt {
    pub bolt: bolt::Bolt,
    pub strokes: Strokes,
    /// The followed camera's position when the bolt struck, metre frame, m.
    pub view: Vec3,
    /// Seconds since the first stroke at the start of this frame; negative when the strike falls
    /// inside the frame that spawned it.
    pub age: f64,
    /// Envelope of every stroke averaged over this frame (the main channel's brightness).
    pub mean: f32,
    /// Envelope of the first stroke alone averaged over this frame (the branches' brightness).
    pub mean_first: f32,
}

impl ActiveBolt {
    /// The attachment point.
    pub(crate) fn ground(&self) -> Vec3 {
        self.bolt.channels[0].points[0]
    }
}

struct PendingThunder {
    /// `Time<Real>` seconds at which the thunder is heard.
    due: f64,
    thunder: Thunder,
}

/// The strike process, the scripted-strike counter and the thunder still in the air.
#[derive(Resource, Default)]
pub(crate) struct Strikes {
    process: Option<StrikeProcess>,
    scripted: u64,
    pending: Vec<PendingThunder>,
}

/// The messages the strike system reads and writes.
#[derive(SystemParam)]
pub(crate) struct StrikeIo<'w, 's> {
    scripted: MessageReader<'w, 's, StrikeLightning>,
    strikes: MessageWriter<'w, LightningStrike>,
    thunder: MessageWriter<'w, Thunder>,
}

/// Both clocks: strikes follow `Time<Virtual>`, thunder `Time<Real>`.
#[derive(SystemParam)]
pub(crate) struct Clocks<'w> {
    virtual_time: Res<'w, Time<Virtual>>,
    real_time: Res<'w, Time<Real>>,
}

/// Advance the strike process, fire scripted strikes and deliver thunder.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_strikes(
    clocks: Clocks,
    weather: Res<Weather>,
    units: Res<ElementsUnits>,
    climate: Res<Climate>,
    view: Res<ElementsView>,
    snapshot: Res<EnvironmentSnapshot>,
    mut strikes: ResMut<Strikes>,
    mut io: StrikeIo,
    mut commands: Commands,
) {
    let dt = clocks.virtual_time.delta_secs_f64();
    let start = clocks.virtual_time.elapsed_secs_f64() - dt;
    let now_real = clocks.real_time.elapsed_secs_f64();
    if strikes
        .process
        .as_ref()
        .is_none_or(|p| p.seed != climate.seed)
    {
        strikes.process = Some(StrikeProcess::new(climate.seed));
    }
    let mut draws = Vec::new();
    if let Some(process) = strikes.process.as_mut() {
        let rate = f64::from(weather.storminess.max(0.0)) / 60.0;
        process.advance(start, dt, rate, &mut draws);
    }
    // (attachment point, bolt seed, age of the first stroke at the start of this frame)
    let mut fired: Vec<(Vec3, u64, f64)> = draws
        .iter()
        .map(|d| {
            let xz = Vec2::new(view.position.x + d.offset.x, view.position.z + d.offset.y);
            let at = Vec3::new(xz.x, snapshot.ground_height(xz), xz.y);
            (at, d.seed, start - d.time)
        })
        .collect();
    // The process works in the metre frame; a scripted strike's `at` is in world units.
    for strike in io.scripted.read() {
        let n = strikes.scripted;
        strikes.scripted += 1;
        fired.push((
            units.point_to_metres(strike.at),
            hash64(climate.seed ^ DOMAIN, (1 << 63) | n),
            0.0,
        ));
    }
    let base_m = CloudLayer::from_weather(&weather).base_m;
    for (position, seed, age) in fired {
        // Messages carry world positions; the distance is in metres.
        let world = units.point_to_world(position);
        io.strikes.write(LightningStrike {
            position: world,
            seed,
        });
        let distance_m = position.distance(view.position);
        strikes.pending.push(PendingThunder {
            due: now_real + f64::from(distance_m / SPEED_OF_SOUND_MS),
            thunder: Thunder {
                position: world,
                distance_m,
                loudness: 1.0 / (1.0 + distance_m / 1000.0),
            },
        });
        commands.spawn(ActiveBolt {
            bolt: bolt::generate(seed, position, base_m),
            strokes: Strokes::new(seed),
            view: view.eye,
            age,
            mean: 0.0,
            mean_first: 0.0,
        });
    }
    strikes.pending.retain(|p| {
        let due = p.due <= now_real;
        if due {
            io.thunder.write(p.thunder);
        }
        !due
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strike_times(seed: u64, rate: f64, horizon: f64) -> Vec<Draw> {
        let mut process = StrikeProcess::new(seed);
        let mut out = Vec::new();
        // One frame per 1/60 s, as the headless test does.
        let dt = 1.0 / 60.0;
        let mut t = 0.0;
        while t < horizon {
            process.advance(t, dt, rate, &mut out);
            t += dt;
        }
        out.retain(|d| d.time <= horizon);
        out
    }

    #[test]
    fn the_stream_reproduces_the_prototype() {
        // docs/research/lightning.md: seed 42's first four strikes and the first strike's offset.
        let draws = strike_times(42, 0.1, 600.0);
        let times: Vec<f64> = draws.iter().take(4).map(|d| d.time).collect();
        for (got, want) in times.iter().zip([11.951, 15.450, 19.188, 38.584]) {
            assert!((got - want).abs() < 1e-3, "{times:?}");
        }
        assert!((draws[0].offset.x - -1174.5).abs() < 0.1, "{:?}", draws[0]);
        assert!((draws[0].offset.y - 7352.9).abs() < 0.1, "{:?}", draws[0]);
        assert_eq!(draws[0].seed, 0x2d57_3357_dae4_3c17);
        let radii: Vec<f32> = draws.iter().take(3).map(|d| d.offset.length()).collect();
        for (got, want) in radii.iter().zip([7446.1, 3800.3, 5663.6]) {
            assert!((got - want).abs() < 0.2, "{radii:?}");
        }
    }

    #[test]
    fn the_hash_matches_the_prototypes_vectors() {
        assert_eq!(hash64(0, 1), 0xE220_A839_7B1D_CDAF);
        assert_eq!(hash64(42, 0), 0xA759_EA27_D472_7622);
    }

    #[test]
    fn counts_over_a_hundred_seeds_are_poisson() {
        let counts: Vec<usize> = (0..100)
            .map(|s| strike_times(s, 0.1, 600.0).len())
            .collect();
        let mean = counts.iter().sum::<usize>() as f64 / 100.0;
        assert!((mean - 60.0).abs() < 2.5, "mean {mean}");
        let inside = counts.iter().filter(|c| (40..=80).contains(*c)).count();
        assert!(inside >= 95, "{inside} of 100 in [40, 80]");
    }

    #[test]
    fn a_ramped_rate_integrates_to_the_expected_count() {
        // Storminess 0 -> 6 over 60 s then 6 for 540 s: 0.1 * (30 + 540) = 57 strikes.
        let counts: Vec<usize> = (0..200)
            .map(|seed| {
                let mut process = StrikeProcess::new(seed);
                let mut out = Vec::new();
                let dt = 1.0 / 60.0;
                for frame in 0..36_000u32 {
                    let t = f64::from(frame) * dt;
                    let storminess = (6.0 * (t + dt) / 60.0).min(6.0);
                    process.advance(t, dt, storminess / 60.0, &mut out);
                }
                out.len()
            })
            .collect();
        let mean = counts.iter().sum::<usize>() as f64 / 200.0;
        assert!((mean - 57.0).abs() < 2.0, "mean {mean}");
        assert!(counts.iter().all(|c| (32..=82).contains(c)), "{counts:?}");
    }

    #[test]
    fn positions_fill_the_disc_uniformly() {
        let mut process = StrikeProcess::new(2024);
        let mut draws = Vec::new();
        // Rate 1000/s for 100 s of one-second frames: 1e5 strikes.
        for frame in 0..100 {
            process.advance(f64::from(frame), 1.0, 1000.0, &mut draws);
        }
        let n = draws.len();
        assert!((99_000..=101_000).contains(&n), "{n}");
        let mut radii: Vec<f32> = draws.iter().map(|d| d.offset.length()).collect();
        assert!(radii.iter().all(|r| *r <= STRIKE_RADIUS_M + 0.01));
        let mean = radii.iter().sum::<f32>() / n as f32;
        assert!((mean / 5333.3 - 1.0).abs() < 0.01, "mean {mean}");
        radii.sort_by(f32::total_cmp);
        let median = radii[n / 2];
        assert!((median / 5656.9 - 1.0).abs() < 0.01, "median {median}");
        let within = |r: f32| radii.iter().filter(|x| **x < r).count() as f64 / n as f64;
        assert!(
            (within(1000.0) / 0.015_625 - 1.0).abs() < 0.1,
            "{}",
            within(1000.0)
        );
        assert!((within(343.0) - 0.0018).abs() < 0.0005, "{}", within(343.0));
        // Bearings: chi-square on 8 bins passes at the 1 % level (critical value 18.48 for 7 dof).
        let mut bins = [0f64; 8];
        for d in &draws {
            let a = d
                .offset
                .y
                .atan2(d.offset.x)
                .rem_euclid(std::f32::consts::TAU);
            bins[((a / std::f32::consts::TAU * 8.0) as usize).min(7)] += 1.0;
        }
        let expected = n as f64 / 8.0;
        let chi: f64 = bins.iter().map(|b| (b - expected).powi(2) / expected).sum();
        assert!(chi < 18.48, "chi-square {chi}");
    }

    #[test]
    fn drawn_strikes_land_on_the_terrain() {
        use std::sync::Arc;
        use std::time::Duration;

        use bevy::time::TimeUpdateStrategy;

        use crate::lightning::LightningPlugin;
        use crate::plugin::ElementsSystems;
        use crate::terrain::{TerrainGrid, TerrainSnapshot};

        #[derive(Resource, Default)]
        struct Seen(Vec<LightningStrike>);

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, LightningPlugin))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_nanos(
                16_666_667,
            )))
            .insert_resource(Climate::temperate().with_seed(42))
            .insert_resource(Weather {
                storminess: 6.0,
                ..default()
            })
            // A flat terrain 250 m up over the whole 8 km strike disc.
            .insert_resource(EnvironmentSnapshot {
                terrain: Some(TerrainSnapshot {
                    grid: TerrainGrid {
                        origin: Vec2::splat(-10_000.0),
                        cell_m: 625.0,
                        size: UVec2::splat(32),
                        y0: 250.0,
                    },
                    surface: Arc::from(vec![0.0f32; 1024]),
                    sediment: Arc::from(vec![0.0f32; 1024]),
                    version: 1,
                }),
                ..default()
            })
            .init_resource::<Seen>()
            .add_systems(
                Update,
                (|mut strikes: MessageReader<LightningStrike>, mut seen: ResMut<Seen>| {
                    seen.0.extend(strikes.read().copied());
                })
                .after(ElementsSystems::Simulation),
            );
        for _ in 0..36_000 {
            app.update();
        }
        let seen = &app.world().resource::<Seen>().0;
        assert!(seen.len() >= 40, "{} strikes", seen.len());
        assert!(
            seen.iter().all(|s| s.position.y == 250.0),
            "{:?}",
            seen.iter().find(|s| s.position.y != 250.0)
        );
    }
}
