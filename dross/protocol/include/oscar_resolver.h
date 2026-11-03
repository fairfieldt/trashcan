#ifndef OSCAR_RESOLVER_H
#define OSCAR_RESOLVER_H
#include "oscar_datagram.h"
#define OSCAR_RESOLVER_LOOKUP_RIGHT 1u
#define OSCAR_QUERY_COLLECT_RIGHT 1u
#define OSCAR_QUERY_OBSERVE_RIGHT 2u
#define OSCAR_QUERY_CANCEL_RIGHT 4u
#define OSCAR_QUERY_INFO_RIGHT 8u
#define OSCAR_QUERY_ALL_RIGHTS 15u
#define OSCAR_QUERY_PENDING 0u
#define OSCAR_QUERY_COMPLETE 1u
#define OSCAR_QUERY_FAILED 2u
#define OSCAR_QUERY_CANCELED 3u
#define OSCAR_QUERY_A 1u
#define OSCAR_QUERY_AAAA 28u
static inline int oscar_resolver_name_valid(const uint8_t*p,size_t n){
 if(!p||!n||n>OSCAR_MAX_RESOLVER_NAME_BYTES)return 0;if(n==1&&p[0]=='.')return 1;if(p[n-1]=='.')n--;if(!n||n>253)return 0;size_t label=0;
 for(size_t i=0;i<n;i++){uint8_t c=p[i];if(c=='.'){if(!label)return 0;label=0;}else{if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='-'||c=='_')||++label>63)return 0;}}
 return label!=0;
}
typedef struct{oscar_header header;uint64_t deadline,subscription,revision,key;uint32_t rights,mask;uint16_t query_type;const uint8_t*name;size_t name_bytes;}oscar_resolver_request;
static inline int oscar_resolver_request_decode(const uint8_t*p,size_t n,size_t handles,oscar_resolver_request*out){
 if(!p||!out)return -1;*out=(oscar_resolver_request){0};int f=oscar_frame_decode(p,n,handles,0,1,&out->header);if(f)return f;oscar_header*h=&out->header;if(h->kind!=OSCAR_KIND_REQUEST)return 1;if(h->body_bytes<8)return -1;const uint8_t*b=p+32;size_t size=h->body_bytes;out->deadline=oscar_get64(b);
 if(h->protocol==OSCAR_PROTOCOL_RESOLVER){
 if(h->ordinal==OSCAR_RESOLVER_DESCRIBE)return size==8&&!handles?0:-1;
 if(h->ordinal!=OSCAR_RESOLVER_LOOKUP)return 1;if(size<16||handles)return -1;out->query_type=oscar_get16(b+8);if(out->query_type!=1&&out->query_type!=28)return 1;out->name_bytes=oscar_get16(b+10);out->rights=oscar_get32(b+12);out->name=b+16;return size==16+out->name_bytes&&!(out->rights&~15u)&&oscar_resolver_name_valid(out->name,out->name_bytes)?0:-1;
 }
 if(h->protocol!=OSCAR_PROTOCOL_RESOLVE_QUERY)return 1;
 if(h->ordinal==1||h->ordinal==3)return size==8&&!handles?0:-1;
 if(h->ordinal>=4&&h->ordinal<=7){oscar_network_request common;int result=oscar_network_observer_request_decode(p,n,handles,OSCAR_PROTOCOL_RESOLVE_QUERY,&common);if(!result){out->subscription=common.subscription;out->revision=common.revision;out->key=common.key;out->mask=common.mask;}return result;}
 return 1;
}
typedef struct{uint32_t rights,state;int32_t error;uint64_t revision,expires_at;}oscar_query_info;
static inline int oscar_query_info_decode(const uint8_t*p,size_t n,oscar_query_info*out){
 if(!p||!out||n!=32||oscar_get32(p))return -1;out->rights=oscar_get32(p+4);out->state=oscar_get32(p+8);out->error=(int32_t)oscar_get32(p+12);out->revision=oscar_get64(p+16);out->expires_at=oscar_get64(p+24);if((out->rights&~15u)||!out->revision)return -1;
 switch(out->state){case 0:return out->error||out->expires_at?-1:0;case 1:return out->error?-1:0;case 2:return out->error>=0||out->error==OSCAR_STATUS_CANCELED||out->expires_at?-1:0;case 3:return out->error!=OSCAR_STATUS_CANCELED||out->expires_at?-1:0;default:return -1;}
}
typedef struct{int32_t status;uint16_t query_type,count;uint32_t ttl_seconds;uint64_t expires_at;const uint8_t*addresses;}oscar_query_answers;
/* Addresses are checked endpoint values with port and scope zero. A repeat
 * collection keeps expires_at; consumers must age ttl_seconds before caching. */
static inline int oscar_query_collect_decode(const uint8_t*p,size_t n,size_t handles,uint16_t expected,oscar_query_answers*out){
 if(!p||!out||n<4||handles)return -1;*out=(oscar_query_answers){0};out->status=(int32_t)oscar_get32(p);if(out->status>0)return -1;if(out->status)return n==4?0:-1;
 if(n<24||oscar_get32(p+12))return -1;out->query_type=oscar_get16(p+4);if(out->query_type!=1&&out->query_type!=28)return 1;out->count=oscar_get16(p+6);out->ttl_seconds=oscar_get32(p+8);out->expires_at=oscar_get64(p+16);if(out->query_type!=expected||!out->count||out->count>OSCAR_MAX_RESOLVER_RECORDS||n!=24+(size_t)out->count*24)return -1;
 out->addresses=p+24;for(size_t i=0;i<out->count;i++){oscar_endpoint e;if(oscar_network_endpoint_decode(p+24+i*24,24,&e)||e.port||e.scope||e.family!=(expected==1?4:6))return -1;}return 0;
}
#endif
