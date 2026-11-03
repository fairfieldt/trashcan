/* Fixed-storage ownership helpers; wire encoding comes from oscar_protocol.h. */
#ifndef DROSS_C_TEST_MESSAGE_H
#define DROSS_C_TEST_MESSAGE_H
#include "native.h"

enum { MESSAGE_CAPACITY=1024, HANDLE_CAPACITY=64 };
typedef struct {
    uint8_t bytes[MESSAGE_CAPACITY];
    handle_t handles[HANDLE_CAPACITY];
    size_t length;
    size_t handle_count;
} owned_message;

/* Every successful receive immediately owns all returned handles, even if its
 * frame or body later fails validation. Zeroed slots mean ownership was moved. */
static inline int message_close(owned_message *message) {
    int ok=1;
    for (size_t i=0;i<HANDLE_CAPACITY;i++) {
        if (close_handle(&message->handles[i])) ok=0;
    }
    message->length=0;
    message->handle_count=0;
    return ok;
}
static inline int64_t message_receive(handle_t channel, owned_message *message,
                                       uint64_t deadline) {
    if (!message_close(message)) return -2;
    native_result result=receive(channel,message->bytes,sizeof(message->bytes),
                                 message->handles,HANDLE_CAPACITY,deadline);
    if (result.status) return result.status;
    message->length=(uint32_t)result.value;
    message->handle_count=(size_t)(result.value>>32);
    if (message->length>MESSAGE_CAPACITY || message->handle_count>HANDLE_CAPACITY) {
        (void)message_close(message);
        return -5;
    }
    return 0;
}
static inline handle_t message_take(owned_message *message, size_t index) {
    if (index>=message->handle_count) return 0;
    handle_t handle=message->handles[index];
    message->handles[index]=0;
    return handle;
}
#endif
