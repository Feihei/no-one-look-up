//! Dedicated input action constants.
//!
//! Never scatter magic key strings/codes through gameplay systems; gameplay
//! code references these constants only (root AGENTS.md, code convention #6).
//! The kinematic controller (R3.4) reads these via `ButtonInput<KeyCode>`.
#![allow(dead_code)] // consumed per milestone: R3 controller/seasons, R4 menus

use bevy::input::keyboard::KeyCode;

// Movement
pub const MOVE_FORWARD: KeyCode = KeyCode::KeyW;
pub const MOVE_BACKWARD: KeyCode = KeyCode::KeyS;
pub const TURN_LEFT: KeyCode = KeyCode::KeyA;
pub const TURN_RIGHT: KeyCode = KeyCode::KeyD;

// Walk speed (in-game adjustable, 1.2–2.0 m/s; no sprint, no jump)
pub const SPEED_UP: KeyCode = KeyCode::Equal;
pub const SPEED_DOWN: KeyCode = KeyCode::Minus;

// Menu / system
pub const PAUSE: KeyCode = KeyCode::Escape;

// Development shortcuts (stripped or gated in release builds)
pub const DEV_TOGGLE_GIZMOS: KeyCode = KeyCode::KeyG;
pub const DEV_SEASON_SPRING: KeyCode = KeyCode::Digit1;
pub const DEV_SEASON_SUMMER: KeyCode = KeyCode::Digit2;
pub const DEV_SEASON_AUTUMN: KeyCode = KeyCode::Digit3;
pub const DEV_SEASON_WINTER: KeyCode = KeyCode::Digit4;
