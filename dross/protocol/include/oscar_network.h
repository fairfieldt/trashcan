#ifndef OSCAR_NETWORK_H
#define OSCAR_NETWORK_H
#include "oscar_protocol.h"
/* Native service rights, deliberately separate from kernel handle rights. */
#define OSCAR_NETWORK_CONNECT_RIGHT 1u
#define OSCAR_NETWORK_LISTEN_RIGHT 2u
#define OSCAR_NETWORK_DATAGRAM_RIGHT 4u
#define OSCAR_STREAM_READ_RIGHT 1u
#define OSCAR_STREAM_WRITE_RIGHT 2u
#define OSCAR_STREAM_OBSERVE_RIGHT 4u
#define OSCAR_STREAM_INFO_RIGHT 8u
#define OSCAR_STREAM_SHUTDOWN_SEND_RIGHT 16u
#define OSCAR_STREAM_SHUTDOWN_RECEIVE_RIGHT 32u
#define OSCAR_STREAM_ABORT_RIGHT 64u
#define OSCAR_STREAM_ALL_RIGHTS 127u
#define OSCAR_LISTENER_ACCEPT_RIGHT 1u
#define OSCAR_LISTENER_OBSERVE_RIGHT 2u
#define OSCAR_LISTENER_INFO_RIGHT 4u
#define OSCAR_LISTENER_ALL_RIGHTS 7u
#define OSCAR_NET_READABLE 1u
#define OSCAR_NET_WRITABLE 2u
#define OSCAR_NET_CONNECTED 4u
#define OSCAR_NET_ACCEPTABLE 8u
#define OSCAR_NET_PEER_CLOSED 16u
#define OSCAR_NET_ERROR 32u
#define OSCAR_NET_COMPLETE 64u
#define OSCAR_NET_DATAGRAM_SIGNALS 51u
#define OSCAR_NET_QUERY_SIGNALS 112u
#define OSCAR_NET_STREAM_SIGNALS 55u
#define OSCAR_NET_LISTENER_SIGNALS 56u
#define OSCAR_NETWORK_MAX_BACKLOG 4u
/* Values only. These structures are never copied to or from the wire. All
 * input/output pointers must reference their stated lengths. Decoding does not
 * take attachment ownership: the enclosing message owner closes rejected caps. */
typedef struct {uint16_t family,port;uint32_t scope;uint8_t address[16];} oscar_endpoint;
static inline int oscar_network_endpoint_decode(const uint8_t*p,size_t n,oscar_endpoint*out){
 if(!p||!out||n!=24)return -1;out->family=oscar_get16(p);out->port=oscar_get16(p+2);out->scope=oscar_get32(p+4);
 if(out->family!=4&&out->family!=6)return -1;
 if(out->family==4){if(out->scope)return -1;for(size_t i=12;i<24;i++)if(p[i])return -1;}
 for(size_t i=0;i<16;i++)out->address[i]=p[8+i];return 0;
}
static inline int oscar_network_endpoint_encode(uint8_t*p,size_t n,const oscar_endpoint*in){
 if(!p||!in||n<24||(in->family!=4&&in->family!=6))return -1;
 if(in->family==4){if(in->scope)return -1;for(size_t i=4;i<16;i++)if(in->address[i])return -1;}
 oscar_put16(p,in->family);oscar_put16(p+2,in->port);oscar_put32(p+4,in->scope);for(size_t i=0;i<16;i++)p[8+i]=in->address[i];return 0;
}
static inline uint32_t oscar_network_signal_mask(uint32_t protocol){return protocol==OSCAR_PROTOCOL_TCP_STREAM?OSCAR_NET_STREAM_SIGNALS:protocol==OSCAR_PROTOCOL_TCP_LISTENER?OSCAR_NET_LISTENER_SIGNALS:protocol==OSCAR_PROTOCOL_DATAGRAM_SOCKET?OSCAR_NET_DATAGRAM_SIGNALS:protocol==OSCAR_PROTOCOL_RESOLVE_QUERY?OSCAR_NET_QUERY_SIGNALS:0;}
static inline uint32_t oscar_network_rights_mask(uint32_t protocol){return protocol==OSCAR_PROTOCOL_NETWORK?7u:protocol==OSCAR_PROTOCOL_TCP_STREAM?127u:protocol==OSCAR_PROTOCOL_TCP_LISTENER?7u:(protocol==OSCAR_PROTOCOL_DATAGRAM_SOCKET||protocol==OSCAR_PROTOCOL_RESOLVE_QUERY)?15u:0;}
typedef struct {oscar_header header;uint64_t deadline,subscription,revision,key;oscar_endpoint endpoint;uint32_t rights,stream_rights,backlog,maximum,mask;const uint8_t*data;size_t data_bytes;} oscar_network_request;
/* 0 valid, 1 unsupported protocol/version/ordinal, -1 malformed. */
static inline int oscar_network_request_decode(const uint8_t*p,size_t n,size_t handles,oscar_network_request*out){
 if(!p||!out)return -1;*out=(oscar_network_request){0};oscar_header*h=&out->header;
 int frame=oscar_frame_decode(p,n,handles,0,1,h);if(frame)return frame;
 if(h->kind!=OSCAR_KIND_REQUEST)return 1;if(h->protocol!=OSCAR_PROTOCOL_NETWORK&&h->protocol!=OSCAR_PROTOCOL_TCP_STREAM&&h->protocol!=OSCAR_PROTOCOL_TCP_LISTENER)return 1;
 if(h->body_bytes<8)return -1;out->deadline=oscar_get64(p+32);const uint8_t*b=p+40;size_t size=h->body_bytes-8;size_t want_handles=0;
 if(h->protocol==OSCAR_PROTOCOL_NETWORK){
  if(h->ordinal==OSCAR_NETWORK_DESCRIBE){if(size)return -1;}
  else if(h->ordinal==OSCAR_NETWORK_CONNECT){if(size!=32||oscar_get32(b+28)||oscar_network_endpoint_decode(b,24,&out->endpoint)||!out->endpoint.port)return -1;out->rights=oscar_get32(b+24);if(out->rights&~127u)return -1;}
  else if(h->ordinal==OSCAR_NETWORK_LISTEN){if(size!=40||oscar_get32(b+36)||oscar_network_endpoint_decode(b,24,&out->endpoint))return -1;out->rights=oscar_get32(b+24);out->stream_rights=oscar_get32(b+28);out->backlog=oscar_get32(b+32);if((out->rights&~7u)||(out->stream_rights&~127u)||!out->backlog||out->backlog>4)return -1;}
  else return 1;
 }else if(h->protocol==OSCAR_PROTOCOL_TCP_LISTENER&&h->ordinal==OSCAR_TCP_LISTENER_ACCEPT){if(size!=8||oscar_get32(b+4))return -1;out->rights=oscar_get32(b);if(out->rights&~127u)return -1;}
 else if(h->protocol==OSCAR_PROTOCOL_TCP_STREAM&&h->ordinal==OSCAR_TCP_STREAM_READ){if(size!=8||oscar_get32(b+4))return -1;out->maximum=oscar_get32(b);if(!out->maximum||out->maximum>OSCAR_MAX_NETWORK_BYTES)return -1;}
 else if(h->protocol==OSCAR_PROTOCOL_TCP_STREAM&&h->ordinal==OSCAR_TCP_STREAM_WRITE){if(size<8||oscar_get32(b+4))return -1;out->data_bytes=oscar_get32(b);if(out->data_bytes>OSCAR_MAX_NETWORK_BYTES||size!=8+out->data_bytes)return -1;out->data=b+8;}
 else if(h->ordinal==4){if(size)return -1;}
 else if(h->ordinal==5){if(size!=16||oscar_get32(b+4))return -1;out->mask=oscar_get32(b);out->key=oscar_get64(b+8);if(!out->mask||(out->mask&~oscar_network_signal_mask(h->protocol)))return -1;want_handles=1;}
 else if(h->ordinal==6){if(size!=16)return -1;out->subscription=oscar_get64(b);out->revision=oscar_get64(b+8);if(!out->subscription||!out->revision)return -1;}
 else if(h->ordinal==7){if(size!=8)return -1;out->subscription=oscar_get64(b);if(!out->subscription)return -1;}
 else if(h->protocol==OSCAR_PROTOCOL_TCP_STREAM&&(h->ordinal==3||h->ordinal==8||h->ordinal==9)){if(size)return -1;}
 else return 1;
 return handles==want_handles?0:-1;
}
typedef struct {uint64_t subscription,key,revision;uint32_t signals;int32_t error;} oscar_network_readiness;
static inline int oscar_network_readiness_valid(const oscar_network_readiness*e){return e&&e->subscription&&e->revision&&!(e->signals&~127u)&&e->error<=0&&((e->error!=0)==((e->signals&OSCAR_NET_ERROR)!=0));}
static inline int oscar_network_event_encode(uint8_t*p,size_t n,const oscar_network_readiness*e){
 if(!p||n<64||!oscar_network_readiness_valid(e))return -1;oscar_header h={OSCAR_PROTOCOL_NETWORK_OBSERVER,1,OSCAR_KIND_EVENT,OSCAR_NETWORK_OBSERVER_READY,32,0,0,0};if(oscar_frame_encode(p,&h))return -1;
 oscar_put64(p+32,e->subscription);oscar_put64(p+40,e->key);oscar_put64(p+48,e->revision);oscar_put32(p+56,e->signals);oscar_put32(p+60,(uint32_t)e->error);return 0;
}
static inline int oscar_network_event_decode(const uint8_t*p,size_t n,size_t handles,oscar_network_readiness*out){
 if(!p||!out)return -1;oscar_header h;if(oscar_frame_decode(p,n,handles,OSCAR_PROTOCOL_NETWORK_OBSERVER,1,&h)||h.kind!=OSCAR_KIND_EVENT||h.ordinal!=OSCAR_NETWORK_OBSERVER_READY||h.body_bytes!=32||handles)return -1;
 out->subscription=oscar_get64(p+32);out->key=oscar_get64(p+40);out->revision=oscar_get64(p+48);out->signals=oscar_get32(p+56);out->error=(int32_t)oscar_get32(p+60);return oscar_network_readiness_valid(out)?0:-1;
}
typedef struct {uint32_t rights,signals;int32_t error;uint64_t revision;uint32_t address_flags;oscar_endpoint local,peer;} oscar_network_info;
static inline int oscar_network_info_decode(uint32_t protocol,const uint8_t*p,size_t n,oscar_network_info*out){
 if(!p||!out||n!=80||oscar_get32(p)||oscar_get32(p+76)||(oscar_get32(p+72)&~3u)||(protocol!=OSCAR_PROTOCOL_TCP_STREAM&&protocol!=OSCAR_PROTOCOL_TCP_LISTENER&&protocol!=OSCAR_PROTOCOL_DATAGRAM_SOCKET))return -1;
 *out=(oscar_network_info){0};out->rights=oscar_get32(p+4);out->signals=oscar_get32(p+8);out->error=(int32_t)oscar_get32(p+12);out->revision=oscar_get64(p+16);out->address_flags=oscar_get32(p+72);
 if((out->rights&~oscar_network_rights_mask(protocol))||(out->signals&~oscar_network_signal_mask(protocol))||out->error>0||!out->revision||((out->error!=0)!=((out->signals&OSCAR_NET_ERROR)!=0)))return -1;
 if(out->address_flags&1){if(oscar_network_endpoint_decode(p+24,24,&out->local))return -1;}else for(size_t i=24;i<48;i++)if(p[i])return -1;
 if(out->address_flags&2){if(oscar_network_endpoint_decode(p+48,24,&out->peer))return -1;}else for(size_t i=48;i<72;i++)if(p[i])return -1;
 return 0;
}
/* Read replies distinguish DATA from EOF; zero requested length is invalid.
 * Nonzero status is a native service result, never a parser result or errno. */
typedef struct {int32_t status;int eof;const uint8_t*data;uint32_t bytes;} oscar_network_read;
static inline int oscar_network_read_decode(const uint8_t*p,size_t n,size_t handles,uint32_t maximum,oscar_network_read*out){
 if(!p||!out||n<4||handles||!maximum||maximum>OSCAR_MAX_NETWORK_BYTES)return -1;*out=(oscar_network_read){0};out->status=(int32_t)oscar_get32(p);if(out->status>0)return -1;if(out->status)return n==4?0:-1;
 if(n<16||oscar_get32(p+12))return -1;uint32_t kind=oscar_get32(p+4);out->bytes=oscar_get32(p+8);if(out->bytes>maximum||n!=16+(size_t)out->bytes)return -1;
 if(kind==0&&out->bytes){out->data=p+16;return 0;}if(kind==1&&!out->bytes){out->eof=1;return 0;}return -1;
}
static inline int oscar_network_write_decode(const uint8_t*p,size_t n,size_t handles,size_t offered,int32_t*status,uint32_t*accepted){
 if(!p||!status||!accepted||n<4||handles||offered>OSCAR_MAX_NETWORK_BYTES)return -1;*status=(int32_t)oscar_get32(p);*accepted=0;if(*status>0)return -1;if(*status)return n==4?0:-1;
 if(n!=8)return -1;*accepted=oscar_get32(p+4);return *accepted<=offered&&(!offered||*accepted)?0:-1;
}
static inline int oscar_network_grant_decode(const uint8_t*p,size_t n,size_t handles,uint32_t requested,int32_t*status,uint32_t*granted){
 if(!p||!status||!granted||n<4)return -1;*status=(int32_t)oscar_get32(p);*granted=0;if(*status>0)return -1;if(*status)return n==4&&!handles?0:-1;
 if(n!=8||handles!=1)return -1;*granted=oscar_get32(p+4);return (*granted&~requested)?-1:0;
}
#endif
