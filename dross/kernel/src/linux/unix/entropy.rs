//! Fail-closed CSPRNG. Test boots supply fresh host entropy; hardware may use RDRAND.
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use spin::Mutex;
static RNG: Mutex<Option<ChaCha20Rng>> = Mutex::new(None);
pub fn initialize(seed: Option<crate::memory::Memory>) {
    let mut bytes = [0; 32];
    if let Some(seed) = seed {
        seed.read(0, &mut bytes).expect("entropy seed");
    }
    let hardware = crate::cpu::cpuid(1, 0).2 & (1 << 30) != 0;
    assert!(
        hardware || bytes.iter().any(|&b| b != 0),
        "Unix needs fresh secure entropy"
    );
    if hardware {
        for chunk in bytes.chunks_mut(8) {
            let mut value = 0u64;
            let mut success = 0u8;
            for _ in 0..1024 {
                // SAFETY: CPUID above confirmed RDRAND. Failure is retried, never replaced by time.
                unsafe {
                    core::arch::asm!("rdrand {}","setc {}",out(reg)value,out(reg_byte)success);
                }
                if success != 0 {
                    break;
                }
            }
            assert!(success != 0, "hardware entropy unavailable");
            for (dst, src) in chunk.iter_mut().zip(value.to_le_bytes()) {
                *dst ^= src;
            }
        }
    }
    *RNG.lock() = Some(ChaCha20Rng::from_seed(bytes));
    bytes.fill(0);
}
pub fn fill(bytes: &mut [u8]) -> Result<(), super::Errno> {
    match RNG.lock().as_mut() {
        Some(rng) => {
            rng.fill_bytes(bytes);
            Ok(())
        }
        None => Err(super::EAGAIN),
    }
}
