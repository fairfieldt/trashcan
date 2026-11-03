#include "oscar_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint8_t bytes[65536];
static size_t unhex(const char*s){size_t n=strlen(s);if(n%2||n/2>sizeof bytes)exit(2);for(size_t i=0;i<n/2;i++){unsigned b;if(sscanf(s+i*2,"%2x",&b)!=1)exit(2);bytes[i]=(uint8_t)b;}return n/2;}
static void hex(const uint8_t*b,size_t n){for(size_t i=0;i<n;i++)printf("%02x",b[i]);puts("");}
int main(int argc,char**argv){if(argc<3)return 2;size_t n=unhex(argv[2]);size_t handles=argc>3?strtoul(argv[3],0,10):0;
 if(!strcmp(argv[1],"endpoint")){oscar_endpoint e;int result=oscar_network_endpoint_decode(bytes,n,&e);if(result){puts("-1");return 0;}uint8_t out[24];if(oscar_network_endpoint_encode(out,sizeof out,&e))return 2;hex(out,24);}
 else if(!strcmp(argv[1],"event")){oscar_network_readiness e;int result=oscar_network_event_decode(bytes,n,handles,&e);if(result){puts("-1");return 0;}uint8_t out[64];if(oscar_network_event_encode(out,sizeof out,&e))return 2;hex(out,64);}
 else if(!strcmp(argv[1],"info")){oscar_network_info info;printf("%d\n",oscar_network_info_decode(OSCAR_PROTOCOL_TCP_STREAM,bytes,n,&info));}
 else if(!strcmp(argv[1],"request")){oscar_network_request r;int result=oscar_network_request_decode(bytes,n,handles,&r);printf("%d\n",result);}
 else if(!strcmp(argv[1],"read")){oscar_network_read r;printf("%d\n",oscar_network_read_decode(bytes,n,handles,16384,&r));}
 else if(!strcmp(argv[1],"write")){int32_t status;uint32_t count;printf("%d\n",oscar_network_write_decode(bytes,n,handles,16384,&status,&count));}
 else if(!strcmp(argv[1],"grant")){int32_t status;uint32_t granted;printf("%d\n",oscar_network_grant_decode(bytes,n,handles,127,&status,&granted));}
 else return 2;return 0;
}
