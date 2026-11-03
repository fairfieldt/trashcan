#define _GNU_SOURCE
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <time.h>
#ifdef NDEBUG
#error Filesystem regression checks require assertions
#endif
struct writer {int fd;char value;};
static void *write_shared(void *arg) {struct writer *w=arg;char data[8192];memset(data,w->value,sizeof data);for(int i=0;i<2;i++)assert(write(w->fd,data,sizeof data)==sizeof data);return 0;}
int main(int argc,char **argv) {
    if(argc>1 && !strcmp(argv[1],"expect-unavailable")){int fd=open("/disk/persist/api-renamed",O_RDONLY);assert(fd<0);puts("filesystem identity rejection: ok");return 0;}
    if(argc>1 && !strcmp(argv[1],"seed-verify")){char data[8192];for(int i=0;i<300;i++){char name[80];snprintf(name,sizeof name,"/disk/many/file-%d",i);int fd=open(name,O_RDONLY);assert(fd>=0);assert(read(fd,data,sizeof data)==12 && !memcmp(data,"seed fixture",12));assert(!close(fd));}int fd=open("/disk/extents",O_RDONLY);assert(fd>=0);for(int i=0;i<20;i++){assert(pread(fd,data,sizeof data,i*8192)==sizeof data);for(unsigned j=0;j<sizeof data;j++)assert(data[j]==0);}assert(!close(fd));assert(open("/disk/denied",O_CREAT|O_WRONLY,0600)<0 && errno==EROFS);puts("read-only stored-seed filesystem: ok");return 0;}
    if(argc>1 && !strcmp(argv[1],"integrity-verify")){assert(open("/disk/many/file-0",O_RDONLY)<0 && errno==EIO);puts("detected metadata corruption: EIO");return 0;}
    int hardware=argc>1 && !strncmp(argv[1],"hardware-",9);
    if(!hardware){int large=open("/disk/test/large",O_RDONLY);assert(large>=0);struct stat big;assert(!fstat(large,&big));assert(big.st_size==5368709243LL);unsigned char zeros[123];assert(pread(large,zeros,sizeof zeros,5368709120LL)==sizeof zeros);for(unsigned i=0;i<sizeof zeros;i++)assert(zeros[i]==0);assert(!close(large));}
    int writing=argc==1 || !strcmp(argv[1],"hardware-write") || !strcmp(argv[1],"hardware-rewrite");
    if(argc>1 && !strcmp(argv[1],"hardware-rewrite")){const char *old[]={"/disk/persist/api-renamed","/disk/persist/concurrent"};for(unsigned i=0;i<2;i++)assert(!unlink(old[i]) || errno==ENOENT);assert(!rmdir("/disk/persist") || errno==ENOENT);}
    const char *name="/disk/persist/api-renamed";
    if(!writing){int fd=open(name,O_RDONLY);assert(fd>=0);struct stat st;assert(!fstat(fd,&st));assert(st.st_size==3);assert((st.st_mode&0777)==0640);assert(st.st_atim.tv_nsec==123456789 && st.st_mtim.tv_nsec==987654321);char data[3];assert(read(fd,data,3)==3 && !memcmp(data,"api",3));assert(!fdatasync(fd));assert(!close(fd));puts("filesystem API persistence: ok");return 0;}
    if(hardware)assert(!mkdir("/disk/persist",0700));
    int concurrent=open("/disk/persist/concurrent",O_CREAT|O_EXCL|O_RDWR,0600);assert(concurrent>=0);pthread_t first,second;struct writer a={concurrent,'a'},b={concurrent,'b'};
    assert(!pthread_create(&first,0,write_shared,&a));assert(!pthread_create(&second,0,write_shared,&b));assert(!pthread_join(first,0));assert(!pthread_join(second,0));assert(lseek(concurrent,0,SEEK_CUR)==32768);
    char chunk[8192];unsigned na=0,nb=0;for(unsigned i=0;i<4;i++){assert(pread(concurrent,chunk,sizeof chunk,i*sizeof chunk)==sizeof chunk);for(unsigned j=1;j<sizeof chunk;j++)assert(chunk[j]==chunk[0]);if(chunk[0]=='a')na++;else if(chunk[0]=='b')nb++;else assert(0);}assert(na==2 && nb==2);assert(!fsync(concurrent));assert(!close(concurrent));
    assert(!mkdir("/disk/persist/subdir",0700));int nested=open("/disk/persist/subdir/file",O_CREAT|O_RDWR,0600);if(nested<0)perror("nested create");assert(nested>=0);
    assert(!rename("/disk/persist/subdir","/disk/persist/moved"));assert(write(nested,"nested",6)==6);assert(!close(nested));assert(!unlink("/disk/persist/moved/file"));assert(!rmdir("/disk/persist/moved"));
    int fd=open("/disk/persist/api",O_CREAT|O_EXCL|O_RDWR,0600);assert(fd>=0);
    assert(write(fd,"api-data",8)==8);assert(!fsync(fd));
    assert(!link("/disk/persist/api","/disk/persist/alias"));
    struct stat original,alias;assert(!fstat(fd,&original));assert(!stat("/disk/persist/alias",&alias));assert(original.st_ino==alias.st_ino && original.st_dev==alias.st_dev && original.st_nlink==2 && alias.st_nlink==2);
    assert(!unlink("/disk/persist/api"));assert(!fstat(fd,&original) && original.st_nlink==1);
    assert(!rename("/disk/persist/alias",name));
    assert(pwrite(fd,"x",1,8192)==1);assert(!fdatasync(fd));
    char gap[32];assert(pread(fd,gap,sizeof gap,1024)==sizeof gap);for(unsigned i=0;i<sizeof gap;i++)assert(gap[i]==0);
    assert(!ftruncate(fd,3));assert(!fchmod(fd,0640));assert(!syncfs(fd));
    struct timespec times[2]={{1700000000,123456789},{1700000000,987654321}};assert(!futimens(fd,times));assert(!fstat(fd,&original));assert(original.st_atim.tv_nsec==123456789 && original.st_mtim.tv_nsec==987654321);
    int other=open("/disk/persist/existing",O_CREAT|O_EXCL|O_RDWR,0600);assert(other>=0);assert(write(other,"old",3)==3);
    assert(!rename(name,"/disk/persist/existing"));char old[3];assert(pread(other,old,3,0)==3 && !memcmp(old,"old",3));assert(!fstat(other,&alias) && alias.st_nlink==0);assert(!unlink("/disk/persist/existing"));assert(!fstat(fd,&original) && original.st_nlink==0);
    assert(!fchmod(fd,0600));assert(!futimens(fd,times));assert(!ftruncate(fd,4));assert(pwrite(fd,"!",1,3)==1);assert(pread(fd,old,3,0)==3 && !memcmp(old,"api",3));assert(!fsync(fd));assert(!close(other));
    int saved=open(name,O_CREAT|O_EXCL|O_RDWR,0640);assert(saved>=0);assert(write(saved,"api",3)==3);assert(!futimens(saved,times));assert(!fsync(saved));assert(!close(saved));
    assert(!close(fd));assert(fsync(fd)==-1 && errno==EBADF);
    puts("filesystem API: ok");return 0;
}
