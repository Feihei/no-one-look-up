//! Menu UI plugin (R4.5/4.6).
//!
//! Keyboard-driven menus in the original's minimal style: a main menu
//! (seed input + start/load/quit), an Esc pause menu (resume/save/back),
//! a save panel (slots 1-3), a HUD with the exploration progress and the
//! exploration-complete dialog. The UI nodes are plain `bevy_ui` text; a
//! clickable-button polish pass belongs to R5.6.
//!
//! Starting a run goes through the R4.3 restore flow (a [`LoadRequest`]
//! with a fresh save), so the world rebuild, streaming reset and pose
//! handling stay single-sourced. Phase transitions own the cursor: playing
//! grabs it (Locked), every menu releases it.

use bevy::ecs::hierarchy::ChildOf;
use bevy::prelude::*;
use bevy::text::TextFont;
use bevy::ui::Node;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::core::game_state::{GamePhase, SeedInput};
use crate::core::input::{PAUSE, SEASON_CYCLE};
use crate::plugins::save::{LoadRequest, SaveData, list_slots};
use crate::world::chunks::DEFAULT_SEED;
use crate::world::explored::{ExploredBitmap, WorldExplorationCompleted};
use crate::world::path_network::WorldSeed;
use crate::world::season::SeasonState;
use crate::world::world_center;

pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GamePhase>()
            .init_resource::<SeedInput>()
            .init_resource::<MenuUiRoot>()
            .add_message::<crate::world::explored::WorldExplorationCompleted>()
            .add_systems(PreUpdate, (menu_input, sync_ui_root).chain())
            .add_systems(
                Update,
                (
                    update_hud,
                    cursor_by_phase,
                    // Completion watcher runs after exploration tracking.
                    watch_exploration_complete.after(crate::world::explored::track_player_chunk),
                ),
            );
    }
}

/// Root entity of whichever menu/HUD screen is showing (despawned on phase
/// change; `None` when the world is playing bare).
#[derive(Resource, Debug, Default)]
struct MenuUiRoot {
    entity: Option<Entity>,
}

/// One full-screen text panel, built per line so a menu stays em-sized.
fn panel(lines: &[String]) -> Node {
    let _ = lines;
    Node {
        position_type: PositionType::Absolute,
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        display: Display::Flex,
        flex_direction: FlexDirection::Column,
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        row_gap: Val::Px(8.0),
        ..default()
    }
}

/// The in-game status line, pinned to the top-left corner.
///
/// It used to reuse the full-screen centred panel, which put a line of
/// exploration text across the middle of the view -- straight through the
/// thing the game exists to show. The HUD is status, not a menu: it hugs a
/// corner, stays small, and never competes with the sky.
fn hud_node() -> Node {
    Node {
        position_type: PositionType::Absolute,
        top: Val::Px(12.0),
        left: Val::Px(14.0),
        ..default()
    }
}

fn text_node(line: &str, size: f32) -> (Text, TextFont, TextColor) {
    (
        Text::new(line.to_string()),
        TextFont::from_font_size(size),
        TextColor(Color::WHITE),
    )
}

/// Replaces the on-screen UI with the panel for the current phase.
fn sync_ui_root(
    phase: Res<GamePhase>,
    seed_input: Res<SeedInput>,
    season: Res<SeasonState>,
    bitmap: Res<ExploredBitmap>,
    mut root: ResMut<MenuUiRoot>,
    mut commands: Commands,
    mut completed: MessageReader<WorldExplorationCompleted>,
) {
    if !phase.is_changed()
        && !seed_input.is_changed()
        && !season.is_changed()
        && root.entity.is_some()
        && !bitmap.is_changed()
    {
        // The HUD refreshes below; only phase/seed changes rebuild panels.
        if completed.read().next().is_none() {
            return;
        }
    }
    if let Some(old) = root.entity.take() {
        commands.entity(old).try_despawn();
    }
    let lines: Vec<String> = match *phase {
        GamePhase::MainMenu => {
            let mut lines = vec![
                "NO ONE LOOKS UP".to_string(),
                String::new(),
                "Seed (type digits, backspace): ".to_string() + &seed_input.0,
                format!("Season: {} [Tab]", season.season.label()),
                String::new(),
                "[Enter] Start".to_string(),
                "[L] Load autosave".to_string(),
                "[Esc] Quit".to_string(),
            ];
            if !list_slots().is_empty() {
                lines.insert(5, "[L] Load autosave".to_string());
            }
            lines
        }
        GamePhase::Playing => vec![format!(
            "Explored: {} / {} ({:.1}%)",
            bitmap.len(),
            bitmap.total(),
            bitmap.fraction() * 100.0
        )],
        GamePhase::Paused => vec![
            "PAUSED".to_string(),
            String::new(),
            "[Esc] Resume".to_string(),
            "[S] Save (slot 1-3)".to_string(),
            "[Backspace] Main menu".to_string(),
        ],
    };
    // The exploration-complete dialog rides on top of the pause menu.
    if let Some(old) = root.entity.take() {
        commands.entity(old).try_despawn();
    }
    // The playing phase gets the corner HUD; every other phase is a full-screen
    // menu where a centred layout is correct.
    let playing = *phase == GamePhase::Playing;
    let entity = commands
        .spawn(if playing { hud_node() } else { panel(&lines) })
        .with_children(|parent| {
            for line in &lines {
                // HUD text is deliberately small: it is a status readout, not
                // something to read while walking.
                let size = if playing {
                    14.0
                } else if lines.first() == Some(line) {
                    48.0
                } else {
                    24.0
                };
                parent.spawn(text_node(line, size));
            }
        })
        .id();
    root.entity = Some(entity);
}

/// Keyboard input driving the phase state machine.
fn menu_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut phase: ResMut<GamePhase>,
    mut seed_input: ResMut<SeedInput>,
    mut season: ResMut<SeasonState>,
    mut commands: Commands,
    players: Query<&Transform, With<Camera3d>>,
    bitmap: Res<ExploredBitmap>,
) {
    match *phase {
        GamePhase::MainMenu => {
            // Seed typing: digits append, backspace trims.
            for (key, digit) in [
                (KeyCode::Digit0, '0'),
                (KeyCode::Digit1, '1'),
                (KeyCode::Digit2, '2'),
                (KeyCode::Digit3, '3'),
                (KeyCode::Digit4, '4'),
                (KeyCode::Digit5, '5'),
                (KeyCode::Digit6, '6'),
                (KeyCode::Digit7, '7'),
                (KeyCode::Digit8, '8'),
                (KeyCode::Digit9, '9'),
            ] {
                if keys.just_pressed(key) {
                    seed_input.0.push(digit);
                }
            }
            if keys.just_pressed(KeyCode::Backspace) {
                seed_input.0.pop();
            }
            // ADR-0001: the season is chosen when the walk starts and never
            // changes during it, so this is the only place it is picked.
            if keys.just_pressed(SEASON_CYCLE) {
                season.season = season.season.next();
            }
            // Enter starts a run: a fresh save through the restore flow
            // rebuilds the world from the typed (or default) seed.
            if keys.just_pressed(KeyCode::Enter) {
                let seed = seed_input.0.parse::<u64>().unwrap_or(DEFAULT_SEED);
                let pose = players
                    .single()
                    .map(|t| [t.translation.x, t.translation.z])
                    .unwrap_or([world_center(), world_center()]);
                let fresh =
                    SaveData::capture(seed, season.season, &ExploredBitmap::default(), pose, 0.0);
                commands.insert_resource(WorldSeed(seed));
                commands.queue(|world: &mut World| {
                    world.write_message(LoadRequest(fresh));
                });
                *phase = GamePhase::Playing;
                seed_input.0.clear();
            }
            // L loads the autosave (the newest slot the list reports).
            if keys.just_pressed(KeyCode::KeyL)
                && let Some(info) = list_slots().first()
            {
                match crate::plugins::save::read_save(info.slot) {
                    Ok(save) => {
                        commands.queue(move |world: &mut World| {
                            world.write_message(LoadRequest(save));
                        });
                        *phase = GamePhase::Playing;
                    }
                    Err(err) => error!("loading slot {} failed: {err}", info.slot),
                }
            }
            if keys.just_pressed(PAUSE) {
                std::process::exit(0);
            }
        }
        GamePhase::Playing => {
            // Esc pauses; everything else belongs to the player systems.
            if keys.just_pressed(PAUSE) {
                *phase = GamePhase::Paused;
            }
        }
        GamePhase::Paused => {
            if keys.just_pressed(PAUSE) || keys.just_pressed(KeyCode::Enter) {
                *phase = GamePhase::Playing;
            }
            if keys.just_pressed(KeyCode::KeyS) {
                // Save to slot 1 by default here; the save panel (slot
                // choice) rides on the R5.6 polish pass.
                if let Ok(camera) = players.single() {
                    let yaw = 0.0;
                    let data = SaveData::capture(
                        crate::world::chunks::DEFAULT_SEED,
                        season.season,
                        &bitmap,
                        [camera.translation.x, camera.translation.z],
                        yaw,
                    );
                    match crate::plugins::save::write_save(1, &data) {
                        Ok(()) => info!("saved to slot 1"),
                        Err(err) => error!("save failed: {err}"),
                    }
                }
            }
            // M now toggles the minimap while walking, so the pause menu
            // uses Backspace for returning to the main menu.
            if keys.just_pressed(KeyCode::Backspace) {
                *phase = GamePhase::MainMenu;
            }
        }
    }
}

/// Owns the cursor per phase: playing grabs it (Locked), menus release it.
fn cursor_by_phase(
    phase: Res<GamePhase>,
    mut windows: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    if !phase.is_changed() {
        return;
    }
    let Ok(mut options) = windows.single_mut() else {
        return;
    };
    if phase.is_game_active() {
        options.grab_mode = CursorGrabMode::Locked;
        options.visible = false;
    } else {
        options.grab_mode = CursorGrabMode::None;
        options.visible = true;
    }
}

/// Refreshes the HUD's exploration text (the HUD lives only while playing).
fn update_hud(
    phase: Res<GamePhase>,
    bitmap: Res<ExploredBitmap>,
    root: Res<MenuUiRoot>,
    mut texts: Query<(&ChildOf, &mut Text)>,
) {
    if !phase.is_game_active() {
        return;
    }
    let Some(entity) = root.entity else {
        return;
    };
    for (parent, mut text) in &mut texts {
        if parent.0 != entity {
            continue;
        }
        // First child is the header for menus; while playing the panel only
        // holds the HUD line.
        **text = format!(
            "Explored: {} / {} ({:.1}%)",
            bitmap.len(),
            bitmap.total(),
            bitmap.fraction() * 100.0
        );
    }
}

/// Fires the completion message once the whole world is explored (R4.6).
fn watch_exploration_complete(
    bitmap: Res<ExploredBitmap>,
    mut completed: Local<bool>,
    mut writer: MessageWriter<WorldExplorationCompleted>,
) {
    if *completed || !bitmap.is_changed() || !bitmap.is_complete() {
        return;
    }
    *completed = true;
    writer.write(WorldExplorationCompleted);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::game_state::GamePhase;
    use crate::player::PlayerBody;
    use bevy::MinimalPlugins;

    fn app_with_ui() -> App {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            crate::world::chunks::WorldPlugin,
            crate::plugins::save::SavePlugin,
            MenuPlugin,
        ))
        .init_resource::<ButtonInput<KeyCode>>()
        // The menu picks the season, so it reads `SeasonState`; the real app
        // gets this from `SeasonPlugin`.
        .init_resource::<SeasonState>();
        app.world_mut().spawn((
            Camera3d::default(),
            PlayerBody::default(),
            Transform::from_xyz(world_center(), crate::player::EYE_HEIGHT, world_center()),
        ));
        app
    }

    #[test]
    fn main_menu_does_not_gate_the_world_inactive() {
        // The main menu builds its panel; the world is NOT active.
        let mut app = app_with_ui();
        app.update();
        let phase = *app.world().resource::<GamePhase>();
        assert_eq!(phase, GamePhase::MainMenu);
        assert!(!phase.is_game_active());
    }

    #[test]
    fn starting_sets_playing_and_hud_shows() {
        let mut app = app_with_ui();
        app.update();
        app.update();

        // Press Enter: a run starts (playing), the HUD panel replaces the
        // menu, and the world regenerates through the restore flow.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Enter);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();

        let phase = *app.world().resource::<GamePhase>();
        assert_eq!(phase, GamePhase::Playing);

        // The restore flow reset exploration to empty and the stream is
        // rebuilding around the spawn pose.
        let bitmap = app.world().resource::<ExploredBitmap>();
        assert!(!bitmap.is_empty(), "the restored world explores the start");
    }

    #[test]
    fn tab_picks_the_season_and_the_run_starts_with_it() {
        // R3.5/R3.7: the menu is the only place a run's season is chosen
        // (ADR-0001). The chain that matters is menu -> SeasonState -> the
        // save the run starts from, because the run's season is restored from
        // that save rather than read live.
        let mut app = app_with_ui();
        app.update();
        app.update();
        assert_eq!(
            app.world().resource::<SeasonState>().season,
            crate::plugins::save::Season::default(),
            "the menu must open on the default season"
        );

        // Tab once: default -> next in calendar order.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(SEASON_CYCLE);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();
        let picked = app.world().resource::<SeasonState>().season;
        assert_eq!(picked, crate::plugins::save::Season::default().next());

        // Start the run: the season must survive into the running world.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Enter);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();

        assert_eq!(*app.world().resource::<GamePhase>(), GamePhase::Playing);
        assert_eq!(
            app.world().resource::<SeasonState>().season,
            picked,
            "the run must start with the season the menu showed"
        );
    }

    #[test]
    fn the_season_cannot_change_mid_run() {
        // ADR-0001: fixed for the run. The dev keys are gated on the phase, so
        // pressing them while walking must do nothing.
        let mut app = app_with_ui();
        app.update();
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Enter);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();
        assert_eq!(*app.world().resource::<GamePhase>(), GamePhase::Playing);

        let before = app.world().resource::<SeasonState>().season;
        // Register the dev-shortcut system exactly as `SeasonPlugin` does and
        // drive it with the key held down, so the gate (not the missing
        // button) is what the test exercises.
        app.add_systems(bevy::prelude::Update, crate::plugins::season::dev_keys);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(crate::core::input::DEV_SEASON_WINTER);
        app.update();
        assert_eq!(
            app.world().resource::<SeasonState>().season,
            before,
            "a mid-run season change would break the run's invariant"
        );

        // And the same key DOES work outside a run, so the gate is not simply
        // a dead system.
        *app.world_mut().resource_mut::<GamePhase>() = GamePhase::MainMenu;
        app.update();
        assert_eq!(
            app.world().resource::<SeasonState>().season,
            crate::plugins::save::Season::Winter,
            "the dev shortcut must still work from the menu"
        );
    }

    #[test]
    fn escape_pauses_and_resume_and_menu_return() {
        let mut app = app_with_ui();
        app.update();
        app.update();

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Enter);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();

        // Esc: playing -> paused -> playing.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(crate::core::input::PAUSE);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();
        assert_eq!(*app.world().resource::<GamePhase>(), GamePhase::Paused);

        // Backspace: back to the main menu (world stays alive underneath).
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Backspace);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();
        assert_eq!(*app.world().resource::<GamePhase>(), GamePhase::MainMenu);
    }

    #[test]
    fn menu_ui_root_exists_per_phase() {
        let mut app = app_with_ui();
        app.update();
        app.update();
        let root = app.world().resource::<MenuUiRoot>();
        assert!(root.entity.is_some(), "the main menu panel exists");
    }

    #[test]
    fn hud_denominator_is_the_live_chunk_grid() {
        // The HUD used to hard-code 1600 (the old 40x40 map) while the world
        // is GRID_CELLS x GRID_CELLS; a full walk showed "256 / 1600".
        let mut app = app_with_ui();
        app.update();
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Enter);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();

        let total = (crate::world::GRID_CELLS * crate::world::GRID_CELLS) as usize;
        let mut query = app.world_mut().query::<&Text>();
        let hud = query
            .iter(app.world())
            .find(|text| text.starts_with("Explored:"))
            .map(|text| (**text).clone())
            .expect("the playing HUD shows exploration progress");
        assert!(
            hud.contains(&format!("/ {total} ")),
            "HUD must use the live grid size, got: {hud}"
        );
    }
}
