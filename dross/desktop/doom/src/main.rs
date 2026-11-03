//! Classic Doom's software rasterizer presents through Oscar's native window API.
//! No SDL, X11 or Wayland dependencies. Audio is intentionally disabled upstream.
use dross_rt::{Error, Handle, bootstrap::Bootstrap, sys};
use oscar_desktop::protocol::Input;
use oscar_desktop_native::window::{Event, Window};
use std::{
    ffi::{CString, c_char, c_int},
    sync::Mutex,
    time::Duration,
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 400;
struct Game {
    window: Window,
    keys: [bool; 256],
    delivered: [bool; 256],
    quit: bool,
    frames: u64,
    #[cfg(feature = "acceptance")]
    observer: Option<Handle>,
}
static GAME: Mutex<Option<Game>> = Mutex::new(None);
unsafe extern "C" {
    fn doomgeneric_Create(argc: c_int, argv: *mut *mut c_char);
    fn doomgeneric_Tick();
    static DG_ScreenBuffer: *mut u32;
    #[cfg(feature = "acceptance")]
    fn oscar_doom_state(out: *mut i32) -> c_int;
}
fn key(usage: u8) -> Option<u8> {
    Some(match usage {
        4..=29 => b'a' + usage - 4,
        30..=38 => b'1' + usage - 30,
        39 => b'0',
        40 => 13,
        41 => 27,
        42 => 127,
        43 => 9,
        44 => 0xa2,
        45 => b'-',
        46 => b'=',
        79 => 0xae,
        80 => 0xac,
        81 => 0xaf,
        82 => 0xad,
        0xe0 | 0xe4 => 0xa3,
        0xe1 | 0xe5 => 0xb6,
        0xe2 | 0xe6 => 0xb8,
        58..=67 => 0xbb + usage - 58,
        _ => return None,
    })
}
impl Game {
    fn events(&mut self) {
        for _ in 0..64 {
            match self.window.poll(0) {
                Ok(Some(Event::Close)) => {
                    self.quit = true;
                    break;
                }
                Ok(Some(Event::Focus(false))) => self.keys.fill(false),
                Ok(Some(Event::Input(Input::KeyboardState { modifiers, keys }))) => {
                    self.keys.fill(false);
                    for usage in keys {
                        if let Some(k) = key(usage) {
                            self.keys[k as usize] = true;
                        }
                    }
                    for bit in 0..8 {
                        if modifiers & (1 << bit) != 0
                            && let Some(k) = key(0xe0 + bit)
                        {
                            self.keys[k as usize] = true;
                        }
                    }
                }
                Ok(_) => {}
                Err(Error::ShouldWait | Error::TimedOut) => break,
                Err(_) => {
                    self.quit = true;
                    break;
                }
            }
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn DG_Init() {
    let b = Bootstrap::read(Handle(1)).expect("Doom bootstrap");
    let _ = sys::close(Handle(1));
    let window = Window::open(
        b.handle("window").expect("window capability"),
        WIDTH,
        HEIGHT,
        "Doom",
    )
    .expect("Doom window");
    *GAME.lock().unwrap() = Some(Game {
        window,
        keys: [false; 256],
        delivered: [false; 256],
        quit: false,
        frames: 0,
        #[cfg(feature = "acceptance")]
        observer: b.handle("check"),
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn DG_DrawFrame() {
    let mut game = GAME.lock().unwrap();
    let game = game.as_mut().unwrap();
    game.events();
    #[cfg(feature = "acceptance")]
    if let Some(observer) = game.observer {
        let mut state = [0i32; 5];
        if unsafe { oscar_doom_state(state.as_mut_ptr()) } != 0 {
            let mut bytes = [0u8; 20];
            for (i, value) in state.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
            let _ = sys::channel_write(observer, &bytes, &[]);
        }
    }
    if game.quit || !game.window.can_present() {
        return;
    }
    // SAFETY: DoomGeneric allocates exactly WIDTH*HEIGHT u32s before DG_Init;
    // engine execution and this callback are on one thread.
    let pixels = unsafe { std::slice::from_raw_parts(DG_ScreenBuffer, (WIDTH * HEIGHT) as usize) };
    match game.window.present(pixels) {
        Ok(()) => game.frames += 1,
        Err(Error::ShouldWait) => {}
        Err(_) => game.quit = true,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn DG_SleepMs(ms: u32) {
    std::thread::sleep(Duration::from_millis(ms.min(1000) as u64));
}
#[unsafe(no_mangle)]
pub extern "C" fn DG_GetTicksMs() -> u32 {
    (sys::clock() / 1_000_000) as u32
}
/// # Safety
/// The engine must supply writable pointers to one int and one byte.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn DG_GetKey(pressed: *mut c_int, out: *mut u8) -> c_int {
    let mut game = GAME.lock().unwrap();
    let game = game.as_mut().unwrap();
    game.events();
    for i in 0..256 {
        if game.keys[i] != game.delivered[i] {
            game.delivered[i] = game.keys[i];
            // SAFETY: DoomGeneric passes live output pointers.
            unsafe {
                *pressed = c_int::from(game.keys[i]);
                *out = i as u8;
            }
            return 1;
        }
    }
    0
}
#[unsafe(no_mangle)]
pub extern "C" fn DG_SetWindowTitle(_title: *const c_char) {}
fn main() {
    let mut args: Vec<CString> = std::env::args()
        .map(|a| CString::new(a).expect("argument contains NUL"))
        .collect();
    if !args.iter().any(|a| a.as_bytes() == b"-nosound") {
        args.push(CString::new("-nosound").unwrap());
    }
    let mut pointers: Vec<_> = args.iter().map(|a| a.as_ptr() as *mut c_char).collect();
    // SAFETY: arguments remain owned for the complete engine lifetime.
    unsafe {
        doomgeneric_Create(pointers.len() as c_int, pointers.as_mut_ptr());
    }
    loop {
        unsafe {
            doomgeneric_Tick();
        }
        if GAME.lock().unwrap().as_ref().is_some_and(|g| g.quit) {
            break;
        }
    }
    // Window Drop releases the mapping/channel and asks windowd to remove it.
    drop(GAME.lock().unwrap().take());
}
