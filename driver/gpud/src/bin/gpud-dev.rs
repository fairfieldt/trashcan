//! Opt-in developer launcher. Holds PCI authority across successful, drained
//! driver replacements; its private control socket is separate from GPU rendering.
use anyhow::{Context, Result, anyhow, bail, ensure};
use dross_abi::rights;
use dross_rt::{
    Handle,
    bootstrap::{Bootstrap, Builder},
    pci::Pci,
    sys,
};
use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    os::unix::{
        fs::{OpenOptionsExt, PermissionsExt},
        net::{UnixListener, UnixStream},
    },
    time::Duration,
};
fn native<T>(r: dross_rt::sys::Result<T>) -> Result<T> {
    r.map_err(|e| anyhow!("native operation: {e:?}"))
}
struct Driver {
    process: Handle,
    control: Handle,
    scanout: Option<Handle>,
}
struct Dev {
    pci_handle: Handle,
    pci: Pci,
    framebuffer: Handle,
    input: Option<Handle>,
    bdf: u16,
    images: HashMap<&'static str, Handle>,
    values: HashMap<String, String>,
    driver: Option<Driver>,
    active_app: Option<Handle>,
}
impl Dev {
    fn start(&mut self) -> Result<()> {
        ensure!(self.driver.is_none(), "driver already exists");
        let (control, child_control) = native(sys::channel_create())?;
        let (scanout_child, scanout) = native(sys::channel_create())?;
        let boot = native(
            Builder::default()
                .handle("pci", native(sys::dup(self.pci_handle, rights::ALL))?)
                .handle("supervisor", child_control)
                .handle("display", scanout_child)
                .send(),
        )?;
        let address = format!("0000:{:02x}:{:02x}.{}", self.bdf >> 8, self.bdf >> 3 & 31, self.bdf & 7);
        let process = match sys::spawn_linux(
            self.images["driver"],
            &["gpud", "serve", &address, "--socket", "/run/gpud/gpu1.sock"],
            &["DROSS_BOOTSTRAP=1"],
            Some(boot),
        ) {
            Ok(process) => process,
            Err(error) => {
                for h in [boot, control, scanout] {
                    let _ = sys::close(h);
                }
                bail!("spawn driver: {error:?}");
            }
        };
        self.driver = Some(Driver { process, control, scanout: Some(scanout) });
        let mut status = [0u8; 8];
        let (n, nh) = native(sys::channel_read(control, &mut status, &mut [], sys::after_ms(15000)))?;
        ensure!(n == 8 && nh == 0 && i64::from_le_bytes(status) == 0, "driver ready failed; epoch retained");
        eprintln!("gpud-dev: driver ready");
        Ok(())
    }
    fn app_finished(&mut self) -> Result<()> {
        if let Some(process) = self.active_app {
            match sys::process_wait(process, sys::clock()) {
                Ok(code) => {
                    let _ = sys::close(process);
                    self.active_app = None;
                    ensure!(code == 0, "previous app exited {code}");
                }
                Err(_) => bail!("application still running; driver epoch retained"),
            }
        }
        Ok(())
    }
    fn stop(&mut self) -> Result<()> {
        self.app_finished()?;
        let Some(driver) = &self.driver else {
            return Ok(());
        };
        native(sys::channel_write(driver.control, &1u32.to_le_bytes(), &[]))?;
        let mut status = [0u8; 8];
        let (n, nh) = native(sys::channel_read(driver.control, &mut status, &mut [], sys::after_ms(2000)))?;
        ensure!(n == 8 && nh == 0 && i64::from_le_bytes(status) == 0, "driver busy or quarantined; epoch retained");
        ensure!(
            native(sys::process_wait(driver.process, sys::after_ms(3000)))? == 0,
            "driver exit failed; epoch retained"
        );
        // pcid verifies the transferred successful Process before disabling bus
        // mastering/MSIs and dropping its retained DMA domain. Never force kill.
        native(self.pci.release(self.bdf, driver.process))?;
        let driver = self.driver.take().unwrap();
        for h in [Some(driver.process), Some(driver.control), driver.scanout].into_iter().flatten() {
            let _ = sys::close(h);
        }
        eprintln!("gpud-dev: driver stopped; PCI/DMA epoch released");
        Ok(())
    }
    fn run(&mut self) -> Result<String> {
        self.app_finished()?;
        if self.driver.as_ref().is_some_and(|d| d.scanout.is_none()) {
            self.stop()?;
        }
        if self.driver.is_none() {
            self.start()?;
        }
        let scanout = self.driver.as_mut().unwrap().scanout.take().unwrap();
        let mut boot = Builder::default()
            .handle("scanout", scanout)
            .handle("display", native(sys::dup(self.framebuffer, rights::ALL))?);
        for (role, name) in [("windowd", "windowd-gpu"), ("doom", "doom"), ("vkdraw", "vkdraw")] {
            if let Some(&image) = self.images.get(role) {
                boot = boot.handle(name, native(sys::dup(image, rights::READ | rights::MAP | rights::TRANSFER))?);
            }
        }
        for (name, value) in &self.values {
            boot = boot.value(name, value);
        }
        let boot = native(boot.send())?;
        let process =
            match sys::spawn_linux(self.images["checker"], &["oscar-scanoutcheck"], &["DROSS_BOOTSTRAP=1"], Some(boot))
            {
                Ok(p) => p,
                Err(e) => {
                    let _ = sys::close(boot);
                    bail!("spawn acceptance: {e:?}");
                }
            };
        self.active_app = Some(process);
        let code = native(sys::process_wait(process, sys::after_ms(60000)))
            .context("acceptance did not exit; epoch retained")?;
        let _ = sys::close(process);
        self.active_app = None;
        ensure!(code == 0, "acceptance exited {code}; verify restore logs before restart");
        Ok("ok: native Doom/Vulkan acceptance exited 0; firmware restore acknowledged".into())
    }
    fn play(&mut self) -> Result<String> {
        self.app_finished()?;
        let image = *self.images.get("session").context("native session image not provisioned")?;
        let input = self.input.context("physical seat input not provisioned")?;
        // Discard events queued between sessions. The real seat remains the only
        // input writer; this launcher never synthesizes play-session input.
        for _ in 0..256 {
            let mut bytes = [0u8; 8192];
            let mut handles = [Handle(0); 64];
            match sys::channel_read(input, &mut bytes, &mut handles, 0) {
                Ok((_, nh)) => {
                    for &h in &handles[..nh] {
                        let _ = sys::close(h);
                    }
                }
                Err(dross_rt::Error::ShouldWait | dross_rt::Error::TimedOut) => break,
                Err(error) => bail!("physical seat: {error:?}"),
            }
        }
        if self.driver.as_ref().is_some_and(|driver| driver.scanout.is_none()) {
            self.stop()?;
        }
        if self.driver.is_none() {
            self.start()?;
        }
        let scanout = self.driver.as_mut().unwrap().scanout.take().unwrap();
        let mut boot = Builder::default()
            .value("app", "doom")
            .handle("scanout", scanout)
            .handle("display", native(sys::dup(self.framebuffer, rights::ALL))?)
            .handle("input", native(sys::dup(input, rights::ALL))?);
        for (role, name) in [("windowd", "module/oscar-windowd-gpu"), ("doom", "module/oscar-doom")] {
            boot =
                boot.handle(name, native(sys::dup(self.images[role], rights::READ | rights::MAP | rights::TRANSFER))?);
        }
        for (name, value) in &self.values {
            boot = boot.value(name, value);
        }
        let process = native(dross_rt::bootstrap::spawn(image, "oscar-session", boot, u64::MAX))?;
        self.active_app = Some(process);
        Ok("ok: physical Doom session started; close its window to release it".into())
    }
    fn command(&mut self, request: &str) -> Result<String> {
        let words: Vec<_> = request.split_whitespace().collect();
        match words.as_slice() {
            ["status"] => Ok(format!("driver={} app={}", self.driver.is_some(), self.active_app.is_some())),
            ["stop"] => {
                self.stop()?;
                Ok("ok: driver stopped".into())
            }
            ["restart"] => {
                self.stop()?;
                self.start()?;
                Ok("ok: driver restarted".into())
            }
            ["run"] => self.run(),
            ["play"] => self.play(),
            ["load", role, path] => {
                let role = match *role {
                    "driver" => "driver",
                    "windowd" => "windowd",
                    "checker" => "checker",
                    "doom" => "doom",
                    "vkdraw" => "vkdraw",
                    "session" => "session",
                    _ => bail!("unknown image role"),
                };
                let image = load_image(path)?;
                if let Some(old) = self.images.insert(role, image) {
                    let _ = sys::close(old);
                }
                Ok(format!("ok: {role} image staged"))
            }
            ["set", key, value]
                if ["LD_LIBRARY_PATH", "LD_PRELOAD", "VK_DRIVER_FILES", "GPUD_SOCKET", "iwad"].contains(key) =>
            {
                self.values.insert((*key).into(), (*value).into());
                Ok("ok: launch value updated".into())
            }
            _ => bail!("use status, load ROLE PATH, set KEY VALUE, run, play, restart, or stop"),
        }
    }
}
fn load_image(path: &str) -> Result<Handle> {
    ensure!(path.starts_with("/tmp/"), "stage executable under /tmp");
    let mut bytes = Vec::new();
    fs::File::open(path)?.take((8 << 20) + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() >= 64
            && bytes.len() <= 8 << 20
            && &bytes[..5] == b"\x7fELF\x02"
            && u16::from_le_bytes([bytes[18], bytes[19]]) == 62,
        "expected bounded x86-64 ELF"
    );
    let memory = native(sys::mem_create((bytes.len() as u64).next_multiple_of(4096), 0))?;
    let result = (|| -> Result<Handle> {
        let at = native(sys::vm_map(memory, 0, (bytes.len() as u64).next_multiple_of(4096), 0, 3))?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len());
        }
        native(sys::vm_unmap(at, (bytes.len() as u64).next_multiple_of(4096)))?;
        native(sys::dup(memory, rights::READ | rights::MAP | rights::TRANSFER | rights::DUPLICATE))
    })();
    let _ = sys::close(memory);
    result
}
fn read_request(stream: &mut UnixStream) -> Result<String> {
    stream.set_nonblocking(true)?;
    let deadline = sys::after_ms(2000);
    let mut request = Vec::new();
    loop {
        let mut bytes = [0u8; 256];
        match stream.read(&mut bytes) {
            Ok(0) => break,
            Ok(n) => {
                request.extend_from_slice(&bytes[..n]);
                ensure!(request.len() <= 768, "request too long");
                if request.contains(&b'\n') {
                    break;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                ensure!(sys::clock() < deadline, "request timeout");
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(String::from_utf8(request)?)
}
fn authorize<'a>(request: &'a str, expected: &str) -> Result<&'a str> {
    let (candidate, command) = request.trim().split_once(' ').context("authorization required")?;
    ensure!(candidate.len() == expected.len(), "authorization required");
    let difference = candidate.bytes().zip(expected.bytes()).fold(0u8, |difference, (a, b)| difference | (a ^ b));
    ensure!(difference == 0, "authorization required");
    Ok(command)
}
fn control_token() -> Result<String> {
    let mut bytes = [0u8; 32];
    native(sys::random(&mut bytes)).context("control token entropy")?;
    let token: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    let path = "/run/gpud/dev.token";
    let _ = fs::remove_file(path);
    let mut file = fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?;
    ensure!(file.metadata()?.permissions().mode() & 0o777 == 0o600, "control token requires mode 0600");
    writeln!(file, "{token}")?;
    Ok(token)
}

fn main() -> Result<()> {
    let b = native(Bootstrap::read(Handle(1)))?;
    let _ = sys::close(Handle(1));
    let pci_handle = b.handle("pci").context("PCI authority")?;
    let mut images = HashMap::new();
    for (role, name) in [
        ("driver", "module/gpud"),
        ("windowd", "module/oscar-windowd-gpu"),
        ("checker", "module/oscar-scanoutcheck"),
        ("doom", "module/oscar-doom"),
        ("vkdraw", "module/oscar-vkdraw"),
        ("session", "module/oscar-session"),
    ] {
        if let Some(h) = b.handle(name) {
            images.insert(role, h);
        } else {
            ensure!(matches!(role, "vkdraw" | "session"), "missing {name}");
        }
    }
    let values = b.values.iter().cloned().collect();
    let mut dev = Dev {
        pci_handle,
        pci: Pci::new(pci_handle),
        framebuffer: b.handle("framebuffer").context("firmware display")?,
        input: b.handle("input"),
        bdf: b.value_u64("bdf").unwrap_or(0x600) as u16,
        images,
        values,
        driver: None,
        active_app: None,
    };
    let path = "/run/gpud/dev.sock";
    fs::create_dir_all("/run/gpud")?;
    let _ = fs::remove_file(path);
    let listener = UnixListener::bind(path)?;
    // Unix socket names have no VFS inode on dross. Authority is a fresh bearer
    // token stored in a real 0600 regular file, checked before every command.
    let token = control_token()?;
    dev.start()?;
    eprintln!("gpud-dev: private developer control ready at {path}");
    for connection in listener.incoming() {
        let mut stream = connection?;
        let result = read_request(&mut stream)
            .and_then(|request| authorize(&request, &token).and_then(|command| dev.command(command)));
        let message = match result {
            Ok(message) => message,
            Err(error) => format!("error: {error:#}"),
        };
        stream.set_nonblocking(false)?;
        let _ = writeln!(stream, "{message}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_or_wrong_token_cannot_authorize_a_command() {
        let token = "1234567890abcdef".repeat(4);
        assert!(authorize("restart", &token).is_err());
        assert!(authorize(&format!("{} restart", "0".repeat(64)), &token).is_err());
        assert_eq!(authorize(&format!("{token} restart\n"), &token).unwrap(), "restart");
    }
}
