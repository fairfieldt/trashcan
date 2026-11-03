#ifndef OSCAR_DATAGRAM_H
#define OSCAR_DATAGRAM_H
#include "oscar_network.h"
#define OSCAR_DATAGRAM_SEND_RIGHT 1u
#define OSCAR_DATAGRAM_RECEIVE_RIGHT 2u
#define OSCAR_DATAGRAM_OBSERVE_RIGHT 4u
#define OSCAR_DATAGRAM_INFO_RIGHT 8u
#define OSCAR_DATAGRAM_ALL_RIGHTS 15u
#define OSCAR_DATAGRAM_TRUNCATED 1u
/* Shared INFO/observer requests. Caller retains ownership of all attachments. */
static inline int oscar_network_observer_request_decode(const uint8_t*p,size_t n,size_t handles,uint32_t protocol,oscar_network_request*out){
 if(!p||!out)return -1;*out=(oscar_network_request){0};int f=oscar_frame_decode(p,n,handles,protocol,1,&out->header);if(f)return f;
 if(out->header.kind!=OSCAR_KIND_REQUEST)return 1;if(out->header.body_bytes<8)return -1;out->deadline=oscar_get64(p+32);const uint8_t*b=p+40;size_t size=out->header.body_bytes-8,want=0;
 switch(out->header.ordinal){
 case 4:if(size)return -1;break;
 case 5:if(size!=16||oscar_get32(b+4))return -1;out->mask=oscar_get32(b);out->key=oscar_get64(b+8);if(!out->mask||(out->mask&~oscar_network_signal_mask(protocol)))return -1;want=1;break;
 case 6:if(size!=16)return -1;out->subscription=oscar_get64(b);out->revision=oscar_get64(b+8);if(!out->subscription||!out->revision)return -1;break;
 case 7:if(size!=8)return -1;out->subscription=oscar_get64(b);if(!out->subscription)return -1;break;
 default:return 1;
 }return handles==want?0:-1;
}
static inline int oscar_datagram_request_decode(const uint8_t*p,size_t n,size_t handles,oscar_network_request*out){
 if(!p||!out)return -1;*out=(oscar_network_request){0};int f=oscar_frame_decode(p,n,handles,0,1,&out->header);if(f)return f;oscar_header*h=&out->header;
 if(h->kind!=OSCAR_KIND_REQUEST)return 1;if(h->body_bytes<8)return -1;out->deadline=oscar_get64(p+32);const uint8_t*b=p+32;size_t size=h->body_bytes;
 if(h->protocol==OSCAR_PROTOCOL_NETWORK&&h->ordinal==OSCAR_NETWORK_OPEN_DATAGRAM){if(size!=40||handles)return -1;if(oscar_get32(b+36))return 1;if(oscar_network_endpoint_decode(b+8,24,&out->endpoint))return -1;out->rights=oscar_get32(b+32);return(out->rights&~15u)?-1:0;}
 if(h->protocol!=OSCAR_PROTOCOL_DATAGRAM_SOCKET)return 1;
 switch(h->ordinal){
 case OSCAR_DATAGRAM_SOCKET_SEND_TO:if(size<40||handles||oscar_get32(b+36)||oscar_network_endpoint_decode(b+8,24,&out->endpoint)||!out->endpoint.port)return -1;out->data_bytes=oscar_get32(b+32);if(out->data_bytes>OSCAR_MAX_NETWORK_BYTES||size!=40+out->data_bytes)return -1;out->data=b+40;return 0;
 case OSCAR_DATAGRAM_SOCKET_RECEIVE:if(size!=16||handles||oscar_get32(b+12))return -1;out->maximum=oscar_get32(b+8);return out->maximum<=OSCAR_MAX_NETWORK_BYTES?0:-1;
 case 4:case 5:case 6:case 7:return oscar_network_observer_request_decode(p,n,handles,OSCAR_PROTOCOL_DATAGRAM_SOCKET,out);
 default:return 1;
 }
}
typedef struct{oscar_endpoint sender,local;uint32_t interface,original_bytes,copied_bytes;int truncated;int32_t status;const uint8_t*data;}oscar_datagram_packet;
static inline int oscar_datagram_receive_decode(const uint8_t*p,size_t n,size_t handles,size_t maximum,oscar_datagram_packet*out){
 if(!p||!out||n<4||handles||maximum>OSCAR_MAX_NETWORK_BYTES)return -1;*out=(oscar_datagram_packet){0};out->status=(int32_t)oscar_get32(p);if(out->status>0)return -1;if(out->status)return n==4?0:-1;
 if(n<72||(oscar_get32(p+4)&~1u)||oscar_get32(p+68))return -1;out->original_bytes=oscar_get32(p+8);out->copied_bytes=oscar_get32(p+12);out->truncated=(oscar_get32(p+4)&1)!=0;out->interface=oscar_get32(p+64);
 size_t want=maximum<out->original_bytes?maximum:out->original_bytes;
 if(out->original_bytes>65535||out->copied_bytes!=want||n!=72+want||out->truncated!=(out->copied_bytes<out->original_bytes)||!out->interface||oscar_network_endpoint_decode(p+16,24,&out->sender)||oscar_network_endpoint_decode(p+40,24,&out->local)||!out->local.port||out->sender.family!=out->local.family)return -1;
 out->data=p+72;return 0;
}
static inline int oscar_datagram_send_decode(const uint8_t*p,size_t n,size_t handles,size_t offered,int32_t*status,uint32_t*accepted){
 if(oscar_network_write_decode(p,n,handles,offered,status,accepted))return -1;return *status||*accepted==offered?0:-1;
}
#endif
