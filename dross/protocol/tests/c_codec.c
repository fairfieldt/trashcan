#include <oscar_protocol.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned char bytes[65536];
static size_t unhex(const char*s,unsigned char*out){size_t n=strlen(s)/2;for(size_t i=0;i<n;i++){char v[3]={s[i*2],s[i*2+1],0};out[i]=(unsigned char)strtoul(v,0,16);}return n;}
static void hex(const unsigned char*b,size_t n){for(size_t i=0;i<n;i++)printf("%02x",b[i]);}
int main(int argc,char**argv){(void)argc;
 if(!strcmp(argv[1],"encode")){size_t n=unhex(argv[9],bytes+32);oscar_header h={(uint32_t)strtoul(argv[2],0,10),(uint16_t)strtoul(argv[3],0,10),(uint16_t)strtoul(argv[4],0,10),(uint32_t)strtoul(argv[5],0,10),(uint32_t)n,strtoull(argv[6],0,10),(uint16_t)strtoul(argv[8],0,10),(uint16_t)strtoul(argv[7],0,10)};if(oscar_frame_encode(bytes,&h))return 2;hex(bytes,n+32);puts("");return 0;}
 size_t n=unhex(argv[2],bytes);size_t handles=strtoul(argv[3],0,10);
 if(!strcmp(argv[1],"bootstrap")){oscar_bootstrap_entry e[OSCAR_MAX_BOOTSTRAP_ENTRIES];size_t count;printf("%d\n",oscar_bootstrap_decode(bytes,n,handles,e,&count));return 0;}
 oscar_header h;int r=oscar_frame_decode(bytes,n,handles,OSCAR_PROTOCOL_ECHO,OSCAR_MAJOR_ECHO,&h);if(r<0){puts("-1");return 0;}printf("%d %u %u %u %u %llu %u %u ",r,h.protocol,h.major,h.kind,h.ordinal,(unsigned long long)h.request_id,h.handle_count,h.flags);hex(bytes+32,n-32);puts("");return 0;
}
