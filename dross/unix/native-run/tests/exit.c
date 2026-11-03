/* Native child for the Unix launcher integration test. No libc or Rust runtime. */
#include "native.h"
#include "oscar_protocol.h"

_Noreturn void c_main(uint64_t bootstrap) {
    uint8_t bytes[128];
    handle_t handles[64] = {0};
    handle_t boot = (handle_t)bootstrap;
    native_result result = receive(boot, bytes, sizeof(bytes), handles, 64, deadline_ms(1000));
    size_t count = 0;
    oscar_bootstrap_entry entries[OSCAR_MAX_BOOTSTRAP_ENTRIES];
    if (result.status || (result.value >> 32) != 0
        || oscar_bootstrap_decode(bytes, (uint32_t)result.value, 0, entries, &count)
        || count != 0 || close_handle(&boot)) {
        process_exit(1);
    }
    LOG_LITERAL("native-run: empty bootstrap and native child exit verified");
    process_exit(42);
}
