//! The Unix runtime loads the file; the child uses the native kernel ABI.
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::process::ExitCode;

use dross_rt::deadline::Deadline;
use dross_rt::launch::{self, SpawnOptions};
use dross_rt::owned::Memory;
use dross_rt::protocol::bootstrap::Builder;

fn run(path: &Path) -> Result<i64, String> {
    let mut file = File::open(path).map_err(|e| format!("open image: {e}"))?;
    let bytes = file
        .metadata()
        .map_err(|e| format!("image size: {e}"))?
        .len();
    let mut header = [0u8; 64];
    file.read_exact(&mut header)
        .map_err(|e| format!("read ELF header: {e}"))?;
    if header[..6] != *b"\x7fELF\x02\x01"
        || u16::from_le_bytes(header[16..18].try_into().unwrap()) != 2
        || u16::from_le_bytes(header[18..20].try_into().unwrap()) != 62
    {
        return Err("image must be a static x86-64 ET_EXEC ELF with a native entry point".into());
    }
    let size = bytes.checked_add(4095).ok_or("image is too large")? & !4095;
    let length = usize::try_from(bytes).map_err(|_| "image is too large")?;
    let memory = Memory::create(size, 0).map_err(|e| format!("allocate image: {e:?}"))?;
    let mapping = memory
        .map(0, size, dross_abi::vm::READ | dross_abi::vm::WRITE)
        .map_err(|e| format!("map image: {e:?}"))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("rewind image: {e}"))?;
    // SAFETY: the backing and mapping are fresh, private and never delegated.
    // No other reference or device can access these bytes during the file read.
    // The file fits in the checked page-rounded mapping and usize length.
    let output = unsafe { std::slice::from_raw_parts_mut(mapping.as_mut_ptr().unwrap(), length) };
    file.read_exact(output)
        .map_err(|e| format!("read image: {e}"))?;
    mapping
        .unmap()
        .map_err(|e| format!("unmap image: {:?}", e.error))?;
    let bootstrap = Builder::new()
        .build()
        .map_err(|e| format!("bootstrap: {:?}", e.error))?;
    let child = launch::spawn(
        memory.as_handle(),
        "native-run-child",
        bootstrap,
        SpawnOptions::default(),
    )
    .map_err(|e| format!("launch {:?}: {:?}", e.stage, e.error))?;
    drop(memory);
    child
        .wait_until(Deadline::FOREVER)
        .map_err(|e| format!("wait: {e:?}"))
}

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(image) = args.next() else {
        eprintln!("usage: native-run IMAGE");
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("usage: native-run IMAGE");
        return ExitCode::from(2);
    }
    match run(Path::new(&image)) {
        Ok(code) => ExitCode::from(u8::try_from(code).unwrap_or(125)),
        Err(error) => {
            eprintln!("native-run: {error}");
            ExitCode::from(1)
        }
    }
}
