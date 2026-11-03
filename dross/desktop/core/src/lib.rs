//! Portable desktop policy. No syscalls, device pointers, host APIs, or global state.
#![no_std]

extern crate alloc;

pub mod app;
pub mod canvas;
pub mod chat;
pub mod protocol;
pub mod server;
pub mod shell;
pub mod terminal;
pub mod ui;
