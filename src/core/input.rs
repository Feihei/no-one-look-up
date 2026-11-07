//! Dedicated input action constants.
//!
//! Never scatter magic key strings/codes through gameplay systems; gameplay
//! code references these constants only (root AGENTS.md, code convention #6).
//! The kinematic controller (R3.4) reads these via `ButtonInput<KeyCode>`.
#![allow(dead_code)] // consumed per milestone: R3 seasons, R4 menus

use bevy::input::keyboard::KeyCode;

// Movement
pub const MOVE_FORWARD: KeyCode = KeyCode::KeyW;
pub const MOVE_BACKWARD: KeyCode = KeyCode::KeyS;
pub const TURN_LEFT: KeyCode = KeyCode::KeyA;
pub const TURN_RIGHT: KeyCode = KeyCode::KeyD;

// Walk speed (in-game adjustable, 1.2–2.0 m/s; no sprint, no jump).
// E/Q match the original project.godot (`physical_keycode` 69 / 81).
pub const SPEED_UP: KeyCode = KeyCode::KeyE;
pub const SPEED_DOWN: KeyCode = KeyCode::KeyQ;

// Release / recapture the mouse cursor (original `toggle_mouse`).
pub const TOGGLE_MOUSE: KeyCode = PAUSE;

// Menu / system
pub const PAUSE: KeyCode = KeyCode::Escape;

// Minimap HUD toggle (only while walking, see `plugins::minimap`).
pub const MINIMAP_TOGGLE: KeyCode = KeyCode::KeyM;

// Development shortcuts (stripped or gated in release builds)
pub const DEV_TOGGLE_GIZMOS: KeyCode = KeyCode::KeyG;

// Sky tuning panel (R5.1): F5 opens it, Up/Down select a row, Left/Right
// adjust. Kept off the walk keys so the panel can be used while standing in
// the scene the change applies to.
pub const DEV_TOGGLE_SKY_PANEL: KeyCode = KeyCode::F5;
pub const DEV_PANEL_UP: KeyCode = KeyCode::ArrowUp;
pub const DEV_PANEL_DOWN: KeyCode = KeyCode::ArrowDown;
pub const DEV_PANEL_LEFT: KeyCode = KeyCode::ArrowLeft;
pub const DEV_PANEL_RIGHT: KeyCode = KeyCode::ArrowRight;
/// Main menu: cycles the season for the run about to start (ADR-0001: the
/// season is chosen at the start and is fixed for the whole walk).
pub const SEASON_CYCLE: KeyCode = KeyCode::Tab;

pub const DEV_SEASON_SPRING: KeyCode = KeyCode::Digit1;
pub const DEV_SEASON_SUMMER: KeyCode = KeyCode::Digit2;
pub const DEV_SEASON_AUTUMN: KeyCode = KeyCode::Digit3;
pub const DEV_SEASON_WINTER: KeyCode = KeyCode::Digit4;
