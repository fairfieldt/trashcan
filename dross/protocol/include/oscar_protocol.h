#ifndef OSCAR_PROTOCOL_H
#define OSCAR_PROTOCOL_H
#include "oscar_schema.h"
#include <stddef.h>
/* Values only; this C structure is never copied to/from the wire. */
typedef struct { uint32_t protocol; uint16_t major,kind; uint32_t ordinal,body_bytes; uint64_t request_id; uint16_t handle_count,flags; } oscar_header;
static inline uint16_t oscar_get16(const uint8_t*p){return (uint16_t)p[0]|(uint16_t)p[1]<<8;}
static inline uint32_t oscar_get32(const uint8_t*p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static inline uint64_t oscar_get64(const uint8_t*p){return (uint64_t)oscar_get32(p)|(uint64_t)oscar_get32(p+4)<<32;}
static inline void oscar_put16(uint8_t*p,uint16_t v){p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);}
static inline void oscar_put32(uint8_t*p,uint32_t v){for(unsigned i=0;i<4;i++)p[i]=(uint8_t)(v>>(8*i));}
static inline void oscar_put64(uint8_t*p,uint64_t v){for(unsigned i=0;i<8;i++)p[i]=(uint8_t)(v>>(8*i));}
/* 0 supported, 1 structurally valid but unsupported, -1 malformed. */
static inline int oscar_frame_decode(const uint8_t*p,size_t n,size_t handles,uint32_t protocol,uint16_t major,oscar_header*h){
 if(n<32||n>OSCAR_MAX_MESSAGE_BYTES)return -1;
 h->protocol=oscar_get32(p);h->major=oscar_get16(p+4);h->kind=oscar_get16(p+6);h->ordinal=oscar_get32(p+8);h->body_bytes=oscar_get32(p+12);h->request_id=oscar_get64(p+16);h->handle_count=oscar_get16(p+24);h->flags=oscar_get16(p+26);
 if(!h->protocol||!h->major||h->kind<1||h->kind>3||((h->kind==OSCAR_KIND_EVENT)!=(h->request_id==0))||h->handle_count>OSCAR_MAX_HANDLES||handles!=h->handle_count||h->body_bytes>OSCAR_MAX_MESSAGE_BYTES-32||n!=32+(size_t)h->body_bytes||oscar_get32(p+28))return -1;
 return h->flags||(protocol&&h->protocol!=protocol)||(major&&h->major!=major)?1:0;
}
static inline int oscar_frame_encode(uint8_t*p,const oscar_header*h){
 if(!h->protocol||!h->major||h->kind<1||h->kind>3||((h->kind==OSCAR_KIND_EVENT)!=(h->request_id==0))||h->handle_count>OSCAR_MAX_HANDLES||h->body_bytes>OSCAR_MAX_MESSAGE_BYTES-32)return -1;
 oscar_put32(p,h->protocol);oscar_put16(p+4,h->major);oscar_put16(p+6,h->kind);oscar_put32(p+8,h->ordinal);oscar_put32(p+12,h->body_bytes);oscar_put64(p+16,h->request_id);oscar_put16(p+24,h->handle_count);oscar_put16(p+26,h->flags);oscar_put32(p+28,0);return 0;
}
/* UTF-8 validation without libc; reject NUL/control names. */
static inline int oscar_name_valid(const uint8_t*p,size_t n){
 if(!n||n>OSCAR_MAX_NAME_BYTES)return 0;
 for(size_t i=0;i<n;){uint32_t c=p[i++];unsigned more=0;uint32_t minimum=0;
  if(c<0x80){if(c<32||c==127)return 0;continue;}
  if(c>=0xc2&&c<=0xdf){more=1;c&=31;minimum=0x80;}else if(c>=0xe0&&c<=0xef){more=2;c&=15;minimum=0x800;}else if(c>=0xf0&&c<=0xf4){more=3;c&=7;minimum=0x10000;}else return 0;
  if(more>n-i)return 0;while(more--){uint8_t b=p[i++];if((b&0xc0)!=0x80)return 0;c=(c<<6)|(b&63);}if(c<minimum||c>0x10ffff||(c>=0xd800&&c<=0xdfff)||(c>=0x80&&c<=0x9f))return 0;
 }return 1;
}
typedef struct {uint8_t kind;const uint8_t*name;uint16_t name_bytes;uint32_t protocol;uint16_t attachment;const uint8_t*value;uint32_t value_bytes;} oscar_bootstrap_entry;
static inline int oscar_bytes_equal(const uint8_t*a,size_t an,const uint8_t*b,size_t bn){if(an!=bn)return 0;for(size_t i=0;i<an;i++)if(a[i]!=b[i])return 0;return 1;}
/* All entries/attachments are checked before the caller takes any ownership. */
static inline int oscar_bootstrap_decode(const uint8_t*message,size_t n,size_t handles,oscar_bootstrap_entry entries[OSCAR_MAX_BOOTSTRAP_ENTRIES],size_t*count){
 oscar_header h;if(oscar_frame_decode(message,n,handles,OSCAR_PROTOCOL_BOOTSTRAP,OSCAR_MAJOR_BOOTSTRAP,&h)||h.kind!=OSCAR_KIND_EVENT||h.ordinal!=OSCAR_BOOTSTRAP_START||h.body_bytes<4)return -1;
 const uint8_t*b=message+32;size_t size=h.body_bytes,at=4;size_t entries_n=oscar_get16(b);uint64_t used=0;if(oscar_get16(b+2)||entries_n>OSCAR_MAX_BOOTSTRAP_ENTRIES)return -1;
 for(size_t i=0;i<entries_n;i++){if(size-at<8)return -1;oscar_bootstrap_entry*e=&entries[i];e->kind=b[at];if(b[at+1])return -1;e->name_bytes=oscar_get16(b+at+2);uint32_t data=oscar_get32(b+at+4);at+=8;if(e->name_bytes>size-at)return -1;e->name=b+at;if(!oscar_name_valid(e->name,e->name_bytes))return -1;at+=e->name_bytes;if(data>size-at)return -1;
  for(size_t j=0;j<i;j++)if(oscar_bytes_equal(e->name,e->name_bytes,entries[j].name,entries[j].name_bytes))return -1;
  e->protocol=0;e->attachment=0;e->value=0;e->value_bytes=0;
  if(e->kind==1){if(data!=8)return -1;e->protocol=oscar_get32(b+at);e->attachment=oscar_get16(b+at+4);if(!e->protocol||oscar_get16(b+at+6)||e->attachment>=handles||(used&(UINT64_C(1)<<e->attachment)))return -1;used|=UINT64_C(1)<<e->attachment;}
  else if(e->kind==2){if(data>OSCAR_MAX_VALUE_BYTES)return -1;e->value=b+at;e->value_bytes=data;}else return -1;at+=data;
 }if(at!=size||used!=(handles==64?UINT64_MAX:((UINT64_C(1)<<handles)-1)))return -1;*count=entries_n;return 0;
}
#endif
