/* Native C interoperability fixture. All wire primitives and bootstrap parsing
 * come from the shared protocol header; this program has no libc or heap. */
#include "native.h"
#include "message.h"
#include "oscar_protocol.h"

static owned_message incoming;
static handle_t bootstrap_handle;
static handle_t directory_handle;
static handle_t provider_handle;
static const uint32_t endpoint_rights=0x93; /* READ | WRITE | TRANSFER | WAIT */

static int cleanup(void) {
    int ok=message_close(&incoming);
    if (close_handle(&provider_handle)) ok=0;
    if (close_handle(&directory_handle)) ok=0;
    if (close_handle(&bootstrap_handle)) ok=0;
    return ok;
}
#define FAIL(text) do { LOG_LITERAL(text); (void)cleanup(); process_exit(1); } while (0)

static int encode_request(uint8_t *bytes, uint32_t protocol, uint16_t major,
                          uint32_t ordinal, uint64_t request_id, size_t body_bytes) {
    const oscar_header header={
        .protocol=protocol, .major=major, .kind=OSCAR_KIND_REQUEST,
        .ordinal=ordinal, .body_bytes=(uint32_t)body_bytes,
        .request_id=request_id, .handle_count=0, .flags=0
    };
    return oscar_frame_encode(bytes,&header)==0;
}
static int receive_reply(handle_t endpoint, uint32_t protocol, uint16_t major,
                          uint32_t ordinal, uint64_t request_id,
                          uint64_t deadline) {
    if (message_receive(endpoint,&incoming,deadline)) return 0;
    oscar_header header;
    return oscar_frame_decode(incoming.bytes,incoming.length,incoming.handle_count,
                               protocol,major,&header)==0
        && header.kind==OSCAR_KIND_REPLY && header.ordinal==ordinal
        && header.request_id==request_id;
}
static int connect_service(const uint8_t *name, size_t name_bytes,
                            uint64_t request_id, int denied) {
    uint8_t bytes[OSCAR_MAX_HEADER_BYTES+12+OSCAR_MAX_NAME_BYTES];
    const size_t body_bytes=12+name_bytes;
    const uint64_t deadline=deadline_ms(1500);
    if (!oscar_name_valid(name,name_bytes)
        || !encode_request(bytes,OSCAR_PROTOCOL_DIRECTORY,OSCAR_MAJOR_DIRECTORY,
                             OSCAR_DIRECTORY_CONNECT,request_id,body_bytes)) return 0;
    uint8_t *body=bytes+OSCAR_MAX_HEADER_BYTES;
    oscar_put32(body,OSCAR_PROTOCOL_ECHO);
    oscar_put32(body+4,endpoint_rights);
    oscar_put16(body+8,(uint16_t)name_bytes);
    oscar_put16(body+10,0);
    for (size_t i=0;i<name_bytes;i++) body[12+i]=name[i];
    /* No attachment is sent. A failed send leaves all local capabilities owned. */
    if (send_message(directory_handle,bytes,OSCAR_MAX_HEADER_BYTES+body_bytes,0,0).status
        || !receive_reply(directory_handle,OSCAR_PROTOCOL_DIRECTORY,
                            OSCAR_MAJOR_DIRECTORY,OSCAR_DIRECTORY_CONNECT,
                            request_id,deadline)) return 0;
    body=incoming.bytes+OSCAR_MAX_HEADER_BYTES;
    if (denied) {
        return incoming.length==OSCAR_MAX_HEADER_BYTES+4
            && incoming.handle_count==0
            && (int32_t)oscar_get32(body)==OSCAR_STATUS_DENIED;
    }
    if (incoming.length!=OSCAR_MAX_HEADER_BYTES+16 || incoming.handle_count!=1
        || (int32_t)oscar_get32(body)!=OSCAR_STATUS_OK
        || oscar_get32(body+4)!=OSCAR_PROTOCOL_ECHO) return 0;
    /* Transfer ownership out of the received message. Clearing the slot prevents
     * its cleanup from closing the endpoint that this client now owns. */
    provider_handle=message_take(&incoming,0);
    return provider_handle!=0;
}
static int echo(uint64_t request_id) {
    /* Binary payload includes NUL and non-UTF-8 bytes, not a C string. */
    static const uint8_t payload[]={0,255,128,127,'C',' ','n','a','t','i','v','e',0,42};
    uint8_t bytes[OSCAR_MAX_HEADER_BYTES+sizeof(payload)];
    const uint64_t deadline=deadline_ms(1500);
    if (!encode_request(bytes,OSCAR_PROTOCOL_ECHO,OSCAR_MAJOR_ECHO,
                         OSCAR_ECHO_ECHO,request_id,sizeof(payload))) return 0;
    for (size_t i=0;i<sizeof(payload);i++) bytes[OSCAR_MAX_HEADER_BYTES+i]=payload[i];
    if (send_message(provider_handle,bytes,sizeof(bytes),0,0).status
        || !receive_reply(provider_handle,OSCAR_PROTOCOL_ECHO,OSCAR_MAJOR_ECHO,
                            OSCAR_ECHO_ECHO,request_id,deadline)) return 0;
    return incoming.handle_count==0
        && incoming.length==OSCAR_MAX_HEADER_BYTES+sizeof(payload)
        && oscar_bytes_equal(incoming.bytes+OSCAR_MAX_HEADER_BYTES,sizeof(payload),
                              payload,sizeof(payload));
}

_Noreturn void c_main(uint64_t bootstrap) {
    bootstrap_handle=(handle_t)bootstrap;
    native_result clock=native_call(N_CLOCK,0,0,0,0,0,0);
    if (clock.status || !bootstrap_handle || bootstrap>UINT32_MAX)
        FAIL("invalid native entry or clock");
    if (message_receive(bootstrap_handle,&incoming,deadline_ms(2000)))
        FAIL("bootstrap receive failed");
    oscar_bootstrap_entry entries[OSCAR_MAX_BOOTSTRAP_ENTRIES];
    size_t entry_count=0;
    static const uint8_t directory_name[]="directory";
    if (oscar_bootstrap_decode(incoming.bytes,incoming.length,incoming.handle_count,
                                entries,&entry_count)
        || entry_count!=1 || incoming.handle_count!=1 || entries[0].kind!=1
        || entries[0].protocol!=OSCAR_PROTOCOL_DIRECTORY
        || !oscar_bytes_equal(entries[0].name,entries[0].name_bytes,
                               directory_name,sizeof(directory_name)-1))
        FAIL("typed bootstrap was not one directory capability");
    directory_handle=message_take(&incoming,entries[0].attachment);
    if (!directory_handle || close_handle(&bootstrap_handle) || !message_close(&incoming))
        FAIL("bootstrap ownership transfer failed");

    /* A directory transport capability cannot become a physical-memory resource. */
    native_result physical=native_call(N_MEM_PHYSICAL,directory_handle,0,4096,0,0,0);
    if (!physical.status) {
        handle_t unexpected=(handle_t)physical.value;
        (void)close_handle(&unexpected);
        FAIL("directory unexpectedly granted physical memory authority");
    }
    if (physical.status!=E_WRONG_TYPE) FAIL("directory type check failed");

    if (!connect_service((const uint8_t *)"echo",4,101,0))
        FAIL("allowed directory connection failed");
    /* The directory must respect the requested rights ceiling. Without DUPLICATE
     * on this endpoint even duplicating it with a broader mask must fail. */
    native_result duplicate=native_call(N_DUP,provider_handle,0xff,0,0,0,0);
    if (!duplicate.status) {
        handle_t unexpected=(handle_t)duplicate.value;
        (void)close_handle(&unexpected);
        FAIL("provider endpoint gained duplicate authority");
    }
    if (duplicate.status!=E_ACCESS_DENIED) FAIL("provider rights check failed");
    if (!echo(102)) FAIL("C to native provider binary echo failed");
    if (!connect_service((const uint8_t *)"forbidden",9,103,1))
        FAIL("directory denial returned wrong status or a capability");
    if (!echo(104)) FAIL("allowed endpoint failed after directory denial");
    if (!cleanup()) FAIL("owned handle cleanup failed");
    LOG_LITERAL("native C bootstrap, directory, echo and denial: ok");
    process_exit(0);
}
