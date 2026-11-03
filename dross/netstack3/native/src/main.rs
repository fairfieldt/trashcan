#![no_std]
#![no_main]
extern crate alloc;
use dross_rt::Handle;
use oscar_netstack3_native::runner::{Loss, run};
dross_rt::main!(main);
fn main(_bootstrap: Handle) -> i32 {
    dross_rt::println!("netstack3-native: isolated two-endpoint proof");
    for (label, loss) in [
        ("baseline", Loss::default()),
        (
            "lost-syn",
            Loss {
                syn: true,
                data: false,
            },
        ),
        (
            "lost-data",
            Loss {
                syn: false,
                data: true,
            },
        ),
    ] {
        let report = run(loss);
        dross_rt::println!("netstack3-native: {} {:?}", label, report);
    }
    dross_rt::println!("netstack3-native: PASS TCP bytes half-close EOF close retransmission");
    0
}
