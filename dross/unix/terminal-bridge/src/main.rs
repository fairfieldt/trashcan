//! Shared PTY adapter for Oscar's native terminal byte-stream channel.
use dross_rt::{Handle, bootstrap::Bootstrap, sys};
use std::{
    ffi::CString,
    io::{Read, Write},
    os::fd::FromRawFd,
};
fn main() {
    let bootstrap = Bootstrap::read(Handle(1)).expect("terminal bootstrap");
    let _ = sys::close(Handle(1));
    let channel = bootstrap.handle("terminal").expect("terminal channel");
    let fd = sys::handle_fd(channel).expect("channel byte-stream fd");
    let _ = sys::close(channel);
    let cols = bootstrap.value_u64("columns").unwrap_or(80).clamp(1, 256) as u16;
    let rows = bootstrap.value_u64("rows").unwrap_or(24).clamp(1, 128) as u16;
    // SAFETY: ordinary POSIX PTY lifecycle. Descriptors are checked before ownership transfer.
    unsafe {
        let master = libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY);
        assert!(master >= 0);
        assert_eq!(libc::grantpt(master), 0);
        assert_eq!(libc::unlockpt(master), 0);
        let slave = libc::open(libc::ptsname(master), libc::O_RDWR | libc::O_NOCTTY);
        assert!(slave >= 0);
        let size = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        assert_eq!(libc::ioctl(master, libc::TIOCSWINSZ, &size), 0);
        let shell = CString::new("/bin/sh").unwrap();
        let arg = CString::new("-i").unwrap();
        let child = libc::fork();
        assert!(child >= 0);
        if child == 0 {
            libc::close(master);
            libc::close(fd);
            if libc::setsid() < 0 || libc::ioctl(slave, libc::TIOCSCTTY, 0) < 0 {
                libc::_exit(126);
            }
            for target in 0..3 {
                if libc::dup2(slave, target) < 0 {
                    libc::_exit(126);
                }
            }
            if slave > 2 {
                libc::close(slave);
            }
            libc::execl(
                shell.as_ptr(),
                shell.as_ptr(),
                arg.as_ptr(),
                std::ptr::null::<libc::c_char>(),
            );
            libc::_exit(127);
        }
        libc::close(slave);
        let mut terminal = std::fs::File::from_raw_fd(fd);
        let mut pty = std::fs::File::from_raw_fd(master);
        let mut bytes = [0; 4096];
        'session: loop {
            let mut fds = [
                libc::pollfd {
                    fd,
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: master,
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];
            if libc::poll(fds.as_mut_ptr(), 2, -1) < 0 {
                if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                break;
            }
            for (i, item) in fds.iter().enumerate() {
                if item.revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) == 0 {
                    continue;
                }
                let result = if i == 0 {
                    terminal.read(&mut bytes)
                } else {
                    pty.read(&mut bytes)
                };
                match result {
                    Ok(0) => break 'session,
                    Ok(n) => {
                        if (if i == 0 {
                            pty.write_all(&bytes[..n])
                        } else {
                            terminal.write_all(&bytes[..n])
                        })
                        .is_err()
                        {
                            break 'session;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => break 'session,
                }
            }
        }
        drop(pty);
        let _ = libc::kill(child, libc::SIGHUP);
        let mut status = 0;
        while libc::waitpid(child, &mut status, 0) < 0 {
            if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
                break;
            }
        }
    }
}
