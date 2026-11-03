/* Freestanding libc subset required by lwext4. Allocator is provided by Rust. */
#include <stddef.h>
void *memchr(const void *p, int c, size_t n) { const unsigned char *s=p; while(n--) { if(*s==(unsigned char)c) return (void*)s; s++; } return 0; }
int strcmp(const char *a,const char *b) { while(*a && *a==*b) {a++;b++;} return (unsigned char)*a-(unsigned char)*b; }
int strncmp(const char *a,const char *b,size_t n) { while(n--) { if(*a!=*b || !*a) return (unsigned char)*a-(unsigned char)*b; a++;b++; } return 0; }
size_t strlen(const char *p) { const char *s=p;while(*s)s++;return s-p; }
char *strcpy(char *d,const char *s) { char *r=d;while((*d++=*s++));return r; }
char *strncpy(char *d,const char *s,size_t n) {char *r=d;while(n && *s){*d++=*s++;n--;}while(n--)*d++=0;return r;}
char *strchr(const char *s,int c) {do{if(*s==(char)c)return (char*)s;}while(*s++);return 0;}
void qsort(void *base,size_t n,size_t width,int (*cmp)(const void*,const void*)) {unsigned char *p=base;for(size_t i=1;i<n;i++)for(size_t j=i;j && cmp(p+(j-1)*width,p+j*width)>0;j--)for(size_t k=0;k<width;k++){unsigned char t=p[(j-1)*width+k];p[(j-1)*width+k]=p[j*width+k];p[j*width+k]=t;}}
