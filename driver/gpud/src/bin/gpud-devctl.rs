//! Client for the opt-in developer supervisor. Reads a protected bearer-token file;
//! never prints that token or puts it in a process argument.
use std::{
    fs,
    io::{Read, Write},
    os::unix::{fs::PermissionsExt, net::UnixStream},
};
fn exchange(request: &str) -> std::io::Result<String> {
    let mut stream = UnixStream::connect("/run/gpud/dev.sock")?;
    writeln!(stream, "{request}")?;
    let mut reply = String::new();
    stream.read_to_string(&mut reply)?;
    Ok(reply)
}
fn main() -> std::io::Result<()> {
    let metadata = fs::metadata("/run/gpud/dev.token")?;
    if !metadata.is_file() || metadata.permissions().mode() & 0o777 != 0o600 {
        return Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "control token requires mode 0600"));
    }
    let token = fs::read_to_string("/run/gpud/dev.token")?;
    let token = token.trim();
    if token.len() != 64 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid control token"));
    }
    let command = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if command == "check-auth" {
        let missing = exchange("status")?;
        let wrong = exchange(&format!("{} status", "0".repeat(64)))?;
        let valid = exchange(&format!("{token} status"))?;
        if !missing.starts_with("error: authorization required")
            || !wrong.starts_with("error: authorization required")
            || !valid.starts_with("driver=")
        {
            return Err(std::io::Error::other("control authentication check failed"));
        }
        println!("ok: missing/wrong tokens rejected; valid token accepted");
        return Ok(());
    }
    let reply = exchange(&format!("{token} {command}"))?;
    print!("{reply}");
    if reply.starts_with("error:") {
        std::process::exit(1);
    }
    Ok(())
}
