//! The bolt: a channel polyline in the manner of Reed & Wyvill (1994), the stroke schedule and
//! its frame-integrated light envelope. Pure CPU, deterministic from a 64-bit seed; no ECS.

use bevy::math::Vec3;

use crate::climate::mix64;

/// Segment length, m: `U(5, 15)` (Reed & Wyvill).
pub(crate) const SEGMENT_M: (f32, f32) = (5.0, 15.0);
/// Mean and standard deviation of the polar deviation from the seed axis, degrees. The draw is
/// folded (`|N|`), which moves 2.3 % of the mass.
pub(crate) const DEVIATION_DEG: (f32, f32) = (16.0, 8.0);
/// Branch probability per main-channel segment (plan 2.3).
pub(crate) const MAIN_BRANCH_P: f32 = 0.15;
/// Branch probability per branch segment (Reed & Wyvill Fig. 4).
pub(crate) const BRANCH_BRANCH_P: f32 = 0.025;
/// Branch length in segments, inclusive (Reed & Wyvill Fig. 4).
pub(crate) const BRANCH_SEGMENTS: (u32, u32) = (10, 20);
/// Branch axis angle to its parent segment, rad (Reed & Wyvill Fig. 4).
pub(crate) const BRANCH_ANGLE_RAD: (f32, f32) = (0.18, 0.75);
/// Width and glow ratio of a branch to its parent (Reed & Wyvill Fig. 4).
pub(crate) const BRANCH_RATIO: (f32, f32) = (0.45, 0.6);
/// Deepest channel level: 0 main, 1 branch, 2 twig.
pub(crate) const MAX_LEVEL: u8 = 2;

/// A small deterministic stream: SplitMix64 counter, `mix64` output.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.0)
    }

    /// Uniform in the open interval (0, 1).
    pub(crate) fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub(crate) fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit() as f32
    }

    /// Uniform integer in `lo..=hi`.
    fn int(&mut self, lo: u32, hi: u32) -> u32 {
        (lo + (self.unit() * f64::from(hi - lo + 1)) as u32).min(hi)
    }

    /// Standard normal (Box–Muller, one value per two draws).
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

/// One unfolded polar deviation draw, degrees: `N(16°, 8°)`. Negative for 2.3 % of draws.
pub(crate) fn sample_deviation_deg(rng: &mut Rng) -> f32 {
    DEVIATION_DEG.0 + DEVIATION_DEG.1 * rng.normal() as f32
}

/// `axis` rotated by the polar angle `theta` at azimuth `phi` about itself (both rad).
fn deflect(axis: Vec3, theta: f32, phi: f32) -> Vec3 {
    let (u, v) = axis.any_orthonormal_pair();
    let (sp, cp) = phi.sin_cos();
    (axis * theta.cos() + (u * cp + v * sp) * theta.sin()).normalize()
}

/// One channel of the bolt: the main channel, a branch or a twig.
#[derive(Clone, Debug)]
pub(crate) struct Channel {
    /// 0 main channel, 1 branch, 2 twig.
    pub level: u8,
    /// Core width relative to the main channel, the product of the ratios down the tree.
    pub width: f32,
    /// Glow factor relative to the main channel, likewise.
    pub glow: f32,
    /// Width ratio to the parent channel (1 for the main channel).
    #[cfg(test)]
    pub ratio: f32,
    /// Polar angle of the channel's axis to its parent segment, rad (0 for the main channel).
    #[cfg(test)]
    pub branch_angle: f32,
    /// World-space node positions, ground end first for the main channel.
    pub points: Vec<Vec3>,
}

#[cfg(test)]
impl Channel {
    /// Number of segments.
    pub(crate) fn segments(&self) -> usize {
        self.points.len() - 1
    }
}

/// A bolt: `channels[0]` is the main channel, the rest are branches and twigs.
#[derive(Clone, Debug)]
pub(crate) struct Bolt {
    pub channels: Vec<Channel>,
}

impl Bolt {
    /// The top of the main channel.
    pub(crate) fn top(&self) -> Vec3 {
        self.channels[0].points[self.channels[0].points.len() - 1]
    }

    /// Total segment count.
    #[cfg(test)]
    pub(crate) fn segments(&self) -> usize {
        self.channels.iter().map(Channel::segments).sum()
    }
}

enum Stop {
    /// Grow until a node reaches this height.
    Height(f32),
    /// Grow exactly this many segments.
    Count(u32),
}

/// A branch point: where the child starts and the direction of the parent segment there.
struct Spawn {
    at: Vec3,
    dir: Vec3,
}

/// Grow a channel from `start` with every segment deflected from `axis` (not from the previous
/// segment, which would random-walk off the target). Returns the nodes and the branch points.
fn grow(
    rng: &mut Rng,
    start: Vec3,
    axis: Vec3,
    stop: Stop,
    branch_p: f32,
) -> (Vec<Vec3>, Vec<Spawn>) {
    let mut points = vec![start];
    let mut spawns = Vec::new();
    let mut p = start;
    let mut made = 0u32;
    loop {
        let theta = sample_deviation_deg(rng).abs().to_radians();
        let phi = rng.range(0.0, std::f32::consts::TAU);
        let length = rng.range(SEGMENT_M.0, SEGMENT_M.1);
        let dir = deflect(axis, theta, phi);
        p += dir * length;
        points.push(p);
        made += 1;
        if branch_p > 0.0 && rng.unit() < f64::from(branch_p) {
            spawns.push(Spawn { at: p, dir });
        }
        let done = match stop {
            Stop::Height(top) => p.y >= top,
            Stop::Count(n) => made >= n,
        };
        if done {
            return (points, spawns);
        }
    }
}

/// Generate the bolt that attaches at `ground` and rises to `top_y`.
///
/// The main channel grows upward from the attachment point, so the strike point is exact and the
/// horizontal wander sits at the cloud end. It stops at the first node at or above `top_y`.
pub(crate) fn generate(seed: u64, ground: Vec3, top_y: f32) -> Bolt {
    let mut rng = Rng::new(seed);
    let (points, spawns) = grow(
        &mut rng,
        ground,
        Vec3::Y,
        Stop::Height(top_y),
        MAIN_BRANCH_P,
    );
    let mut channels = vec![Channel {
        level: 0,
        width: 1.0,
        glow: 1.0,
        #[cfg(test)]
        ratio: 1.0,
        #[cfg(test)]
        branch_angle: 0.0,
        points,
    }];
    // Breadth-first over (parent index, spawn): branches, then twigs of the branches.
    let mut pending: Vec<(usize, Spawn)> = spawns.into_iter().map(|s| (0, s)).collect();
    let mut next = 0;
    while next < pending.len() {
        let (parent, spawn) = {
            let (parent, spawn) = &pending[next];
            (
                *parent,
                Spawn {
                    at: spawn.at,
                    dir: spawn.dir,
                },
            )
        };
        next += 1;
        let level = channels[parent].level + 1;
        let ratio = rng.range(BRANCH_RATIO.0, BRANCH_RATIO.1);
        let glow_ratio = rng.range(BRANCH_RATIO.0, BRANCH_RATIO.1);
        let angle = rng.range(BRANCH_ANGLE_RAD.0, BRANCH_ANGLE_RAD.1);
        let azimuth = rng.range(0.0, std::f32::consts::TAU);
        let axis = deflect(spawn.dir, angle, azimuth);
        let count = rng.int(BRANCH_SEGMENTS.0, BRANCH_SEGMENTS.1);
        let p = if level < MAX_LEVEL {
            BRANCH_BRANCH_P
        } else {
            0.0
        };
        let (points, children) = grow(&mut rng, spawn.at, axis, Stop::Count(count), p);
        let index = channels.len();
        channels.push(Channel {
            level,
            width: channels[parent].width * ratio,
            glow: channels[parent].glow * glow_ratio,
            #[cfg(test)]
            ratio,
            #[cfg(test)]
            branch_angle: angle,
            points,
        });
        pending.extend(children.into_iter().map(|s| (index, s)));
    }
    Bolt { channels }
}

/// Envelope time constant, s: the plan's "decays over 0.2 s" read as down to 1 %.
pub(crate) const TAU_S: f64 = 0.2 / 4.605_170_185_988_092;
/// Restrikes: `U{1, 2, 3}`.
pub(crate) const RESTRIKES: (u32, u32) = (1, 3);
/// Restrike spacing, s: `U(40, 80)` ms.
pub(crate) const RESTRIKE_SPACING_S: (f32, f32) = (0.040, 0.080);
/// Restrike amplitude relative to the first stroke: `U(0.1, 0.25)`.
pub(crate) const RESTRIKE_AMPLITUDE: (f32, f32) = (0.1, 0.25);
/// Below this total envelope after the last stroke the flash is over.
pub(crate) const ENVELOPE_END: f64 = 1e-3;

/// The strokes of one flash: the first at `t = 0` with amplitude 1, then 1–3 restrikes.
#[derive(Clone, Debug)]
pub(crate) struct Strokes {
    /// Stroke times, s after the first stroke, ascending.
    pub times: Vec<f64>,
    /// Stroke amplitudes, 1 for the first.
    pub amplitudes: Vec<f64>,
}

impl Strokes {
    /// The schedule for a bolt seed (an independent stream from the bolt's shape).
    pub(crate) fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed ^ 0x5742_4F4B_4553_5452);
        let mut times = vec![0.0];
        let mut amplitudes = vec![1.0];
        for _ in 0..rng.int(RESTRIKES.0, RESTRIKES.1) {
            let last = times[times.len() - 1];
            times.push(last + f64::from(rng.range(RESTRIKE_SPACING_S.0, RESTRIKE_SPACING_S.1)));
            amplitudes.push(f64::from(
                rng.range(RESTRIKE_AMPLITUDE.0, RESTRIKE_AMPLITUDE.1),
            ));
        }
        Self { times, amplitudes }
    }

    /// The time of the last stroke, s.
    pub(crate) fn last(&self) -> f64 {
        self.times[self.times.len() - 1]
    }

    /// The envelope at `t` seconds after the first stroke (1 at the first stroke's peak).
    pub(crate) fn instant(&self, t: f64) -> f64 {
        self.times
            .iter()
            .zip(&self.amplitudes)
            .filter(|(tj, _)| t >= **tj)
            .map(|(tj, a)| a * (-(t - tj) / TAU_S).exp())
            .sum()
    }

    /// The envelope averaged over the frame `[ta, tb]`, so a stroke that falls between two
    /// frames is never skipped and `∫ mean dt = Σ aⱼ τ`.
    pub(crate) fn mean(&self, ta: f64, tb: f64) -> f64 {
        self.mean_of(ta, tb, self.times.len())
    }

    /// As [`Self::mean`] for the first stroke alone (the branches' brightness).
    pub(crate) fn mean_first(&self, ta: f64, tb: f64) -> f64 {
        self.mean_of(ta, tb, 1)
    }

    fn mean_of(&self, ta: f64, tb: f64, strokes: usize) -> f64 {
        if tb <= ta {
            return if strokes == 1 {
                self.amplitudes[0] * self.instant_first(ta)
            } else {
                self.instant(ta)
            };
        }
        let sum: f64 = self
            .times
            .iter()
            .zip(&self.amplitudes)
            .take(strokes)
            .filter(|(tj, _)| tb > **tj)
            .map(|(tj, a)| a * ((-(ta.max(*tj) - tj) / TAU_S).exp() - (-(tb - tj) / TAU_S).exp()))
            .sum();
        TAU_S / (tb - ta) * sum
    }

    fn instant_first(&self, t: f64) -> f64 {
        if t >= 0.0 { (-t / TAU_S).exp() } else { 0.0 }
    }

    /// Whether the flash is over at `t`: past the last stroke with the envelope below
    /// [`ENVELOPE_END`].
    pub(crate) fn finished(&self, t: f64) -> bool {
        t > self.last() && self.instant(t) < ENVELOPE_END
    }
}

/// Horizontal distance of the main channel's top from its attachment point, m.
#[cfg(test)]
pub(crate) fn wander_m(bolt: &Bolt) -> f32 {
    {
        let d = bolt.top() - bolt.channels[0].points[0];
        bevy::math::Vec2::new(d.x, d.z).length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GROUND: Vec3 = Vec3::new(100.0, 0.0, -50.0);
    const TOP: f32 = 1500.0;

    fn bolts(n: u64) -> Vec<Bolt> {
        (0..n)
            .map(|i| generate(crate::climate::hash64(0xB017, i), GROUND, TOP))
            .collect()
    }

    fn mean(values: impl Iterator<Item = f64>) -> f64 {
        let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
        sum / n as f64
    }

    fn segments(channel: &Channel) -> impl Iterator<Item = Vec3> + '_ {
        channel.points.windows(2).map(|w| w[1] - w[0])
    }

    #[test]
    fn main_channel_follows_the_note() {
        let bolts = bolts(300);
        // Mean segment length 10.0 m, U(5, 15).
        let lengths = mean(
            bolts
                .iter()
                .flat_map(|b| b.channels.iter().flat_map(segments))
                .map(|s| f64::from(s.length())),
        );
        assert!((lengths - 10.0).abs() < 0.1, "segment length {lengths}");
        // Polar angle to the seed axis: folded normal, mean 16.13 deg, rms 17.88 deg.
        let angles: Vec<f64> = bolts
            .iter()
            .flat_map(|b| segments(&b.channels[0]))
            .map(|s| f64::from((s.y / s.length()).acos().to_degrees()))
            .collect();
        let mean_angle = mean(angles.iter().copied());
        let rms = mean(angles.iter().map(|a| a * a)).sqrt();
        assert!((mean_angle - 16.13).abs() < 0.2, "mean angle {mean_angle}");
        assert!((rms - 17.88).abs() < 0.2, "rms angle {rms}");
        // 158 main-channel segments for a 1500 m channel.
        let count = mean(bolts.iter().map(|b| b.channels[0].segments() as f64));
        assert!((count - 158.0).abs() < 1.5, "segments {count}");
        // Path length over rise 1.051.
        let (path, rise) = bolts.iter().fold((0.0, 0.0), |(p, r), b| {
            let c = &b.channels[0];
            (
                p + segments(c).map(|s| f64::from(s.length())).sum::<f64>(),
                r + f64::from(c.points[c.points.len() - 1].y - c.points[0].y),
            )
        });
        assert!(
            (path / rise - 1.051).abs() < 0.003,
            "path/rise {}",
            path / rise
        );
        // Mean angle between successive segments 22.6 deg.
        let between = mean(bolts.iter().flat_map(|b| {
            let s: Vec<Vec3> = segments(&b.channels[0]).collect();
            s.windows(2)
                .map(|w| f64::from(w[0].angle_between(w[1]).to_degrees()))
                .collect::<Vec<_>>()
        }));
        assert!((between - 22.6).abs() < 0.5, "successive angle {between}");
    }

    #[test]
    fn deviation_draws_fold_at_two_point_three_percent() {
        let mut rng = Rng::new(1);
        let n = 1_000_000;
        let below = (0..n)
            .filter(|_| sample_deviation_deg(&mut rng) < 0.0)
            .count();
        let fraction = below as f64 / f64::from(n);
        assert!((fraction - 0.0228).abs() < 0.002, "{fraction}");
    }

    #[test]
    fn branches_follow_the_note() {
        let bolts = bolts(300);
        let level = |l: u8| {
            move |b: &Bolt| {
                b.channels
                    .iter()
                    .filter(|c| c.level == l)
                    .map(Channel::segments)
                    .sum::<usize>() as f64
            }
        };
        let count = |l: u8| {
            mean(
                bolts
                    .iter()
                    .map(|b| b.channels.iter().filter(|c| c.level == l).count() as f64),
            )
        };
        let branches = count(1);
        assert!((branches - 23.7).abs() < 1.0, "branches {branches}");
        let branch_segments = mean(bolts.iter().map(level(1)));
        let twig_segments = mean(bolts.iter().map(level(2)));
        let total = mean(bolts.iter().map(|b| b.segments() as f64));
        assert!((branch_segments - 349.0).abs() < 20.0, "{branch_segments}");
        assert!((twig_segments - 130.0).abs() < 15.0, "{twig_segments}");
        assert!((total - 637.0).abs() < 25.0, "{total}");
        for bolt in &bolts {
            for c in &bolt.channels {
                assert!(c.level <= MAX_LEVEL);
                if c.level > 0 {
                    assert!((10..=20).contains(&c.segments()), "{}", c.segments());
                    assert!((0.18..=0.75).contains(&c.branch_angle));
                    assert!((0.45..=0.6).contains(&c.ratio));
                }
            }
            // The attachment point is exact; the cloud end wanders less than 150 m.
            assert_eq!(bolt.channels[0].points[0], GROUND);
            assert!(wander_m(bolt) < 150.0, "{}", wander_m(bolt));
            assert!(bolt.top().y >= TOP);
        }
    }

    #[test]
    fn the_plans_supercritical_branching_is_not_what_ships() {
        // 0.15 per segment with children 0.6x the parent would give 18 578 expected segments at
        // depth 2 for a 150-segment trunk; the adopted generator stays near 637.
        let worst = bolts(1000).iter().map(Bolt::segments).max().unwrap_or(0);
        assert!(worst < 2000, "{worst}");
    }

    #[test]
    fn generation_is_deterministic() {
        let a = generate(7, GROUND, TOP);
        let b = generate(7, GROUND, TOP);
        assert_eq!(a.channels.len(), b.channels.len());
        assert_eq!(a.channels[0].points, b.channels[0].points);
        assert_ne!(
            a.channels[0].points,
            generate(8, GROUND, TOP).channels[0].points
        );
    }

    #[test]
    fn restrikes_follow_the_note() {
        let n = 10_000u64;
        let all: Vec<Strokes> = (0..n)
            .map(|i| Strokes::new(crate::climate::hash64(5, i)))
            .collect();
        for k in 1..=3 {
            let p = all.iter().filter(|s| s.times.len() == k + 1).count() as f64 / n as f64;
            assert!((p - 1.0 / 3.0).abs() < 0.02, "{k} restrikes: {p}");
        }
        let multiplicity = mean(all.iter().map(|s| s.times.len() as f64));
        assert!((multiplicity - 3.0).abs() < 0.02, "{multiplicity}");
        let spacings: Vec<f64> = all
            .iter()
            .flat_map(|s| s.times.windows(2).map(|w| w[1] - w[0]))
            .collect();
        assert!(spacings.iter().all(|d| (0.040..=0.080).contains(d)));
        assert!((mean(spacings.iter().copied()) - 0.060).abs() < 0.0005);
        let geometric = (mean(spacings.iter().map(|d| d.ln()))).exp();
        assert!((geometric - 0.0589).abs() < 0.0005, "{geometric}");
        assert!(all.iter().all(|s| s.amplitudes[0] == 1.0
            && s.amplitudes[1..].iter().all(|a| (0.1..=0.25).contains(a))));
        let length = mean(all.iter().map(|s| s.last() + 0.2));
        assert!((length - 0.32).abs() < 0.005, "flash length {length}");
        assert!(
            all.iter()
                .all(|s| (0.24..=0.44).contains(&(s.last() + 0.2)))
        );
    }

    #[test]
    fn the_envelope_conserves_energy_and_never_skips_a_stroke() {
        let mut rng = Rng::new(99);
        for i in 0..200u64 {
            let strokes = Strokes::new(crate::climate::hash64(3, i));
            let dt = 1.0 / 240.0 + (1.0 / 30.0 - 1.0 / 240.0) * rng.unit();
            let start = -dt * rng.unit();
            let mut t = start;
            let mut integral = 0.0;
            let end = strokes.last() + 3.0;
            let mut lit = vec![false; strokes.times.len()];
            while t < end {
                integral += strokes.mean(t, t + dt) * dt;
                for (j, tj) in strokes.times.iter().enumerate() {
                    if t + dt > *tj
                        && strokes.amplitudes[j]
                            * ((-(t.max(*tj) - tj) / TAU_S).exp() - (-(t + dt - tj) / TAU_S).exp())
                            > 0.0
                    {
                        lit[j] = true;
                    }
                }
                t += dt;
            }
            let expected: f64 = strokes.amplitudes.iter().sum::<f64>() * TAU_S;
            assert!(
                (integral - expected).abs() / expected < 1e-9,
                "{integral} vs {expected}"
            );
            assert!(lit.iter().all(|l| *l));
        }
    }

    #[test]
    fn the_envelope_hits_the_notes_checkpoints() {
        let s = Strokes {
            times: vec![0.0, 0.060],
            amplitudes: vec![1.0, 0.2],
        };
        // 1 % at 0.2 s after the last stroke.
        let last = Strokes {
            times: vec![0.0],
            amplitudes: vec![1.0],
        };
        assert!((last.instant(0.2) - 0.01).abs() < 1e-12);
        // 25.1 % of the first stroke 60 ms later (the restrike adds on top).
        assert!((s.instant(0.060) - (0.2 + 0.251)).abs() < 1e-3);
        // First frame of a 1/60 s frame: stroke at the start, middle, end.
        let dt = 1.0 / 60.0;
        assert!((last.mean(0.0, dt) - 0.830).abs() < 1e-3);
        assert!((last.mean(-0.5 * dt, 0.5 * dt) - 0.455).abs() < 1e-3);
        assert!((last.mean(-0.99 * dt, 0.01 * dt) - 0.010).abs() < 1e-3);
        assert!(last.finished(0.31) && !last.finished(0.29));
    }
}
