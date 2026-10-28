//! Sky tuning panel (R5.1): live, keyboard-driven control of the golden-hour
//! look.
//!
//! F5 opens it; Up/Down picks a row, Left/Right nudges the value. Every change
//! writes `HeadsUpSky`, which the sky systems already treat as the single
//! source of truth, so nothing here is a preview-only path — what you tune in
//! the panel is what the game renders.
//!
//! Why a panel at all: cloud and dusk tuning is iterative and the parameters
//! interact (cover changes the cloud type, humidity changes it again, exposure
//! changes how much of the result you can see). Recompiling between attempts
//! made that loop too slow to do well.
//!
//! Scope: only parameters bevy_elements exposes through `Weather`,
//! `CloudLayer` or the camera are listed. Cloud density, noise scale and march
//! step count are compiled into the crate's shaders, so they are NOT here —
//! changing those means patching the dependency, which is a separate decision.

use bevy::prelude::*;

use crate::core::input::{
    DEV_PANEL_DOWN, DEV_PANEL_LEFT, DEV_PANEL_RIGHT, DEV_PANEL_UP, DEV_TOGGLE_SKY_PANEL,
};
use crate::plugins::sky::HeadsUpSky;

/// Whether the panel is drawn and accepting keys.
#[derive(Resource, Debug, Default)]
pub struct SkyPanelState {
    pub open: bool,
    /// Selected row index into [`PARAMS`].
    pub selected: usize,
}

/// One tunable knob.
pub struct SkyParam {
    /// Row label; the panel prints `name: value`.
    pub name: &'static str,
    /// Reads the current value out of `HeadsUpSky`.
    pub get: fn(&HeadsUpSky) -> f32,
    /// Writes a new value back.
    pub set: fn(&mut HeadsUpSky, f32),
    /// Nudge per Left/Right press, already in the parameter's own units.
    pub step: f32,
    /// Clamp range.
    pub range: (f32, f32),
    /// Decimal places to print.
    pub decimals: usize,
    /// How to write the float: `None` = plain number, `Some(fmt)` = custom.
    pub format: Option<fn(f32) -> String>,
}

/// Every knob the panel exposes, in display order.
///
/// Grouped as the eye sees them: what the clouds are, how big they are, how
/// the light sits, then the photo settings.
pub const PARAMS: &[SkyParam] = &[
    // --- clouds -----------------------------------------------------------
    SkyParam {
        name: "cloud cover",
        // Reads as the preset value until the first nudge sets an override.
        get: |sky| sky.cloud_cover.unwrap_or(0.05),
        set: |sky, v| sky.cloud_cover = Some(v),
        step: 0.02,
        range: (0.0, 1.0),
        decimals: 2,
        format: None,
    },
    SkyParam {
        name: "humidity",
        get: |sky| sky.relative_humidity,
        set: |sky, v| sky.relative_humidity = v,
        step: 0.02,
        range: (0.0, 1.0),
        decimals: 2,
        format: None,
    },
    SkyParam {
        name: "storminess",
        get: |sky| sky.storminess,
        set: |sky, v| sky.storminess = v,
        step: 0.05,
        range: (0.0, 6.0),
        decimals: 2,
        format: None,
    },
    SkyParam {
        name: "wind m/s",
        get: |sky| sky.wind_speed_ms,
        set: |sky, v| sky.wind_speed_ms = v,
        step: 0.5,
        range: (0.0, 25.0),
        decimals: 1,
        format: None,
    },
    // --- cloud layer ------------------------------------------------------
    SkyParam {
        name: "cloud base m",
        get: |sky| sky.cloud_layer.map(|(b, _)| b).unwrap_or(1500.0),
        set: |sky, v| {
            let top = sky.cloud_layer.map(|(_, t)| t).unwrap_or(v + 1800.0);
            sky.cloud_layer = Some((v, top.max(v + 200.0)));
        },
        step: 50.0,
        range: (200.0, 5000.0),
        decimals: 0,
        format: None,
    },
    SkyParam {
        name: "cloud top m",
        get: |sky| sky.cloud_layer.map(|(_, t)| t).unwrap_or(3300.0),
        set: |sky, v| {
            let base = sky.cloud_layer.map(|(b, _)| b).unwrap_or(1500.0);
            sky.cloud_layer = Some((base.min(v - 200.0), v.max(base + 200.0)));
        },
        step: 100.0,
        range: (400.0, 9500.0),
        decimals: 0,
        format: None,
    },
    // --- light ------------------------------------------------------------
    SkyParam {
        name: "exposure ev",
        get: |sky| sky.exposure_ev100,
        set: |sky, v| sky.exposure_ev100 = v,
        step: 0.1,
        range: (8.0, 18.0),
        decimals: 2,
        format: None,
    },
    SkyParam {
        name: "bloom",
        get: |sky| sky.bloom_intensity,
        set: |sky, v| sky.bloom_intensity = v,
        step: 0.02,
        range: (0.0, 1.5),
        decimals: 2,
        format: None,
    },
    SkyParam {
        name: "exp jitter",
        get: |sky| sky.exposure_noise_amplitude,
        set: |sky, v| sky.exposure_noise_amplitude = v,
        step: 0.01,
        range: (0.0, 0.6),
        decimals: 2,
        format: None,
    },
];

/// Marker on the panel's UI root.
#[derive(Component)]
struct SkyPanelRoot;

/// Reads input and updates `HeadsUpSky` / [`SkyPanelState`].
fn sky_panel_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<SkyPanelState>,
    mut sky: ResMut<HeadsUpSky>,
) {
    if keys.just_pressed(DEV_TOGGLE_SKY_PANEL) {
        state.open = !state.open;
        info!(
            "sky panel {}",
            if state.open {
                "open (F5 to close, Up/Down select, Left/Right adjust)"
            } else {
                "closed"
            }
        );
        if !state.open {
            return;
        }
    }
    if !state.open {
        return;
    }

    if keys.just_pressed(DEV_PANEL_UP) {
        state.selected = (state.selected + PARAMS.len() - 1) % PARAMS.len();
    }
    if keys.just_pressed(DEV_PANEL_DOWN) {
        state.selected = (state.selected + 1) % PARAMS.len();
    }

    let mut delta = 0.0;
    if keys.just_pressed(DEV_PANEL_LEFT) {
        delta -= 1.0;
    }
    if keys.just_pressed(DEV_PANEL_RIGHT) {
        delta += 1.0;
    }
    // Held keys repeat: nudging a 50 m step one press at a time is unusable.
    if keys.pressed(DEV_PANEL_LEFT) || keys.pressed(DEV_PANEL_RIGHT) {
        // `just_pressed` already counted the first press; only add repeats
        // after the initial delay so a single tap stays one step.
        if keys.pressed(DEV_PANEL_LEFT) && !keys.just_pressed(DEV_PANEL_LEFT) {
            delta -= 1.0;
        }
        if keys.pressed(DEV_PANEL_RIGHT) && !keys.just_pressed(DEV_PANEL_RIGHT) {
            delta += 1.0;
        }
    }
    if delta == 0.0 {
        return;
    }

    let param = &PARAMS[state.selected];
    let current = (param.get)(&sky);
    let next = (current + delta * param.step).clamp(param.range.0, param.range.1);
    (param.set)(&mut sky, next);
    info!("{} = {}", param.name, format_value(param, next));
}

fn format_value(param: &SkyParam, value: f32) -> String {
    match param.format {
        Some(f) => f(value),
        None => format!("{value:.*}", param.decimals),
    }
}

/// Draws the panel text; the whole panel is rebuilt only when something
/// changed (there are ~9 lines, but rebuilding every frame is still waste).
fn sky_panel_ui(
    state: Res<SkyPanelState>,
    sky: Res<HeadsUpSky>,
    mut commands: Commands,
    roots: Query<Entity, With<SkyPanelRoot>>,
    mut cached: Local<Option<(bool, usize, Vec<String>)>>,
) {
    let lines: Vec<String> = if state.open {
        PARAMS
            .iter()
            .enumerate()
            .map(|(index, param)| {
                let value = format_value(param, (param.get)(&sky));
                let marker = if index == state.selected { ">" } else { " " };
                format!("{marker} {:<14} {}", param.name, value)
            })
            .collect()
    } else {
        Vec::new()
    };

    let signature = (state.open, state.selected, lines.clone());
    if cached.as_ref() == Some(&signature) && !roots.is_empty() {
        return;
    }
    *cached = Some(signature);

    for root in &roots {
        commands.entity(root).try_despawn();
    }
    if !state.open {
        return;
    }

    commands
        .spawn((
            Name::new("SkyPanel"),
            SkyPanelRoot,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(18.0),
                top: Val::Px(18.0),
                padding: UiRect::all(Val::Px(12.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(3.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.05, 0.72)),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("SKY TUNING  F5"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.95, 0.85, 0.55)),
            ));
            for line in &lines {
                panel.spawn((
                    Text::new(line.clone()),
                    TextFont::from_font_size(12.0),
                    TextColor(Color::srgb(0.88, 0.90, 0.92)),
                ));
            }
        });
}

pub struct SkyPanelPlugin;

impl Plugin for SkyPanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SkyPanelState>()
            .init_resource::<HeadsUpSky>()
            .add_systems(Update, (sky_panel_input, sky_panel_ui).chain());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_with_panel() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<SkyPanelState>()
            .init_resource::<HeadsUpSky>()
            .add_systems(Update, sky_panel_input);
        app
    }

    /// Press, run one frame, then release and clear the edge state.
    ///
    /// `MinimalPlugins` has no `InputPlugin`, so nothing calls `clear()` each
    /// frame and `just_pressed` would stay true forever — a second `update()`
    /// would toggle again. In the real game `keyboard_input_system` does this;
    /// here the test has to.
    fn tap(app: &mut App, key: KeyCode) {
        let mut input = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        input.press(key);
        app.update();
        let mut input = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        input.release(key);
        input.clear();
        app.update();
    }

    #[test]
    fn panel_starts_closed_and_ignores_keys_until_opened() {
        // A panel that eats arrow keys while closed would fight the walker
        // (A/D turn, E/Q speed are separate, but Up/Down are also used by
        // future UI).
        let mut app = app_with_panel();
        let before = app.world().resource::<HeadsUpSky>().exposure_ev100;
        tap(&mut app, DEV_PANEL_RIGHT);
        assert_eq!(app.world().resource::<HeadsUpSky>().exposure_ev100, before);
        assert!(!app.world().resource::<SkyPanelState>().open);
    }

    #[test]
    fn f5_toggles_and_arrows_adjust_the_selected_row() {
        let mut app = app_with_panel();
        tap(&mut app, DEV_TOGGLE_SKY_PANEL);
        assert!(app.world().resource::<SkyPanelState>().open);

        // Row 0 is cloud cover. Before any nudge it reads the preset's own
        // value (0.05); one press must move it by exactly one step from THERE,
        // not from the bottom of the range (which would make the first press
        // appear to do nothing).
        let panel = app.world().resource::<SkyPanelState>().selected;
        assert_eq!(panel, 0);
        let default_cover = 0.05;
        tap(&mut app, DEV_PANEL_RIGHT);
        let cover = app
            .world()
            .resource::<HeadsUpSky>()
            .cloud_cover
            .expect("the nudge must write a real override");
        assert!(
            (cover - (default_cover + PARAMS[0].step)).abs() < 1e-4,
            "first nudge should be one step above the preset value, got {cover}"
        );

        // Selecting another row changes which value moves.
        tap(&mut app, DEV_PANEL_DOWN);
        tap(&mut app, DEV_PANEL_DOWN);
        assert_eq!(app.world().resource::<SkyPanelState>().selected, 2);
        let storm_before = app.world().resource::<HeadsUpSky>().storminess;
        tap(&mut app, DEV_PANEL_RIGHT);
        assert!(
            app.world().resource::<HeadsUpSky>().storminess > storm_before,
            "storminess should have increased"
        );
        // And the previous row is untouched.
        assert_eq!(
            app.world().resource::<HeadsUpSky>().cloud_cover,
            Some(cover)
        );
    }

    #[test]
    fn values_stay_inside_their_declared_range() {
        // Hammer every row to both limits: nothing may escape its range, and
        // nothing may become NaN (a NaN exposure blanks the whole frame).
        let mut app = app_with_panel();
        tap(&mut app, DEV_TOGGLE_SKY_PANEL);
        for (row, param) in PARAMS.iter().enumerate() {
            app.world_mut().resource_mut::<SkyPanelState>().selected = row;
            for _ in 0..400 {
                tap(&mut app, DEV_PANEL_RIGHT);
            }
            let high = (param.get)(app.world().resource::<HeadsUpSky>());
            for _ in 0..800 {
                tap(&mut app, DEV_PANEL_LEFT);
            }
            let low = (param.get)(app.world().resource::<HeadsUpSky>());
            let (min, max) = param.range;
            for (value, label) in [(low, "low"), (high, "high")] {
                assert!(
                    value.is_finite(),
                    "{} went non-finite at its {label} end",
                    param.name
                );
                assert!(
                    value >= min - 1e-3 && value <= max + 1e-3,
                    "{} = {value} escaped [{min}, {max}] at its {label} end",
                    param.name
                );
            }
        }
    }

    #[test]
    fn cloud_layer_rows_keep_base_below_top() {
        // `CloudLayer` with top <= base is a degenerate slab; the layer rows
        // must never produce one, whichever end you drag.
        let mut app = app_with_panel();
        tap(&mut app, DEV_TOGGLE_SKY_PANEL);

        let base_row = PARAMS
            .iter()
            .position(|p| p.name == "cloud base m")
            .expect("base row");
        let top_row = PARAMS
            .iter()
            .position(|p| p.name == "cloud top m")
            .expect("top row");

        for row in [base_row, top_row] {
            app.world_mut().resource_mut::<SkyPanelState>().selected = row;
            for _ in 0..200 {
                tap(&mut app, DEV_PANEL_RIGHT);
            }
            if let Some((base, top)) = app.world().resource::<HeadsUpSky>().cloud_layer {
                assert!(top > base, "row {row}: top {top} <= base {base}");
            }
            for _ in 0..400 {
                tap(&mut app, DEV_PANEL_LEFT);
            }
            if let Some((base, top)) = app.world().resource::<HeadsUpSky>().cloud_layer {
                assert!(top > base, "row {row}: top {top} <= base {base}");
            }
        }
    }
}
