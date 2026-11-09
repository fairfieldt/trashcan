//! Native process launch with explicit ownership at each failure stage.
//!
//! This is a direct kernel launch helper, not a supervisor or a service-readiness
//! protocol. Bootstrap bytes may use BootstrapV2 or an existing private format.
//! Only the bootstrap endpoint moves to the child; other capabilities must be
//! attached explicitly to that endpoint's first message.

use crate::owned::{BorrowedHandle, Channel, MemoryObject, Message, Process, Untyped};
use crate::{Error, sys};

/// CPU placement and optional delegated memory account. The kernel checks that
/// `account` is an account with WRITE authority. None creates a child account
/// under the caller's default budget; it does not inherit the parent's handle table.
pub struct SpawnOptions<'a> {
    pub cpu: u64,
    pub account: Option<BorrowedHandle<'a, Untyped>>,
}

impl Default for SpawnOptions<'_> {
    fn default() -> Self {
        Self {
            cpu: u64::MAX,
            account: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchStage {
    Validate,
    CreateBootstrapChannel,
    QueueBootstrap,
    Spawn,
}

/// Before enqueue, ownership remains in Message. After enqueue, ownership is in
/// the child-side channel's queued message. Receive from that channel to recover
/// the capabilities; their original numeric handle values are not preserved.
#[derive(Debug)]
pub enum ReturnedBootstrap {
    Unsent(Message),
    Queued(Channel),
}

#[derive(Debug)]
pub struct LaunchError {
    pub error: Error,
    pub stage: LaunchStage,
    pub bootstrap: ReturnedBootstrap,
}

/// Start a native executable with an explicitly supplied first bootstrap message.
///
/// The image is borrowed and remains owned by the caller. On success the child
/// owns its bootstrap endpoint, and the returned Process observes its lifetime.
/// The child may exit before this function returns. Successful start does not
/// imply the child is ready to provide a service.
///
/// On failure, the returned value owns the unsent message or the queued bootstrap
/// channel. Dropping that value closes its capabilities. The native SPAWN syscall
/// restores the bootstrap endpoint when it fails, so this helper can return it;
/// that guarantee must not be assumed for a future remote launcher protocol.
pub fn spawn(
    image: BorrowedHandle<'_, MemoryObject>,
    name: &str,
    bootstrap: Message,
    options: SpawnOptions<'_>,
) -> Result<Process, LaunchError> {
    // Avoid the raw ABI's truncation in the middle of a UTF-8 process name.
    if name.is_empty() || name.len() > 64 {
        return Err(LaunchError {
            error: Error::InvalidArgs,
            stage: LaunchStage::Validate,
            bootstrap: ReturnedBootstrap::Unsent(bootstrap),
        });
    }
    let (sender, child) = match Channel::pair() {
        Ok(pair) => pair,
        Err(error) => {
            return Err(LaunchError {
                error,
                stage: LaunchStage::CreateBootstrapChannel,
                bootstrap: ReturnedBootstrap::Unsent(bootstrap),
            });
        }
    };
    if let Err(failure) = sender.try_send(bootstrap) {
        return Err(LaunchError {
            error: failure.error,
            stage: LaunchStage::QueueBootstrap,
            bootstrap: ReturnedBootstrap::Unsent(failure.message),
        });
    }
    drop(sender);
    let boot = child.as_handle().as_raw();
    let result = match options.account {
        Some(account) => sys::spawn_in(
            account.as_raw(),
            image.as_raw(),
            name,
            Some(boot),
            options.cpu,
        ),
        None => sys::spawn(image.as_raw(), name, Some(boot), options.cpu),
    };
    match result {
        Ok(process) => {
            // SPAWN consumed the endpoint. Disarm its Rust owner before it drops.
            let _ = child.into_raw();
            // SAFETY: successful SPAWN returns a fresh Process handle.
            Ok(unsafe { Process::from_raw(process) })
        }
        Err(error) => Err(LaunchError {
            error,
            stage: LaunchStage::Spawn,
            bootstrap: ReturnedBootstrap::Queued(child),
        }),
    }
}
