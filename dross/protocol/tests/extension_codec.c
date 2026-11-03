#include "oscar_resolver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint8_t bytes[65536];
static size_t unhex(const char*s){size_t n=strlen(s);if(n%2||n/2>sizeof bytes)exit(2);for(size_t i=0;i<n/2;i++){unsigned b;if(sscanf(s+i*2,"%2x",&b)!=1)exit(2);bytes[i]=(uint8_t)b;}return n/2;}
int main(int argc,char**argv){if(argc<3)return 2;size_t n=unhex(argv[2]);size_t handles=argc>3?strtoul(argv[3],0,10):0;int result;
 if(!strcmp(argv[1],"udp-request")){oscar_network_request r;result=oscar_datagram_request_decode(bytes,n,handles,&r);}
 else if(!strcmp(argv[1],"dns-request")){oscar_resolver_request r;result=oscar_resolver_request_decode(bytes,n,handles,&r);}
 else if(!strcmp(argv[1],"packet")){oscar_datagram_packet r;result=oscar_datagram_receive_decode(bytes,n,handles,3,&r);}
 else if(!strcmp(argv[1],"send")){int32_t status;uint32_t accepted;result=oscar_datagram_send_decode(bytes,n,handles,3,&status,&accepted);}
 else if(!strcmp(argv[1],"answers4")||!strcmp(argv[1],"answers6")){oscar_query_answers r;result=oscar_query_collect_decode(bytes,n,handles,argv[1][7]=='4'?1:28,&r);}
 else if(!strcmp(argv[1],"query-info")){oscar_query_info r;result=oscar_query_info_decode(bytes,n,&r);}
 else if(!strcmp(argv[1],"name")){result=oscar_resolver_name_valid(bytes,n)?0:-1;}
 else if(!strcmp(argv[1],"event")){oscar_network_readiness r;result=oscar_network_event_decode(bytes,n,handles,&r);}
 else return 2;printf("%d\n",result);return 0;
}
