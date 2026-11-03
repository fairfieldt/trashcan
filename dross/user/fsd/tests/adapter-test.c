/* Disposable-image test of the same adapter linked into native fsd. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <assert.h>
static int disk,failed,flush_failed;
static long stop_after=-1,writes;
int fsd_blocks(int write,uint64_t at,uint32_t count,void *buf) {
    if(write && failed)return EIO;
    size_t len=(size_t)count*512;
    if(write && stop_after>=0){if(getenv("TRACE_WRITES"))fprintf(stderr,"write %ld at %llu count %u\n",writes,(unsigned long long)at,count);if(writes++==stop_after)_exit(77);}
    ssize_t done=write?pwrite(disk,buf,len,at*512):pread(disk,buf,len,at*512);
    if(done!=(ssize_t)len){if(write)failed=1;return EIO;}return 0;
}
void fsd_integrity(const char *file,unsigned line){fprintf(stderr,"metadata checksum failed at %s:%u\n",file,line);}
int fsd_flush(void) {return flush_failed || fsync(disk)?EIO:0;}
extern void ext4_dmask_set(uint32_t);
extern int oscar_error(void);
extern int oscar_mount(uint32_t,uint64_t,int),oscar_unmount(void),oscar_sync(void);
extern int oscar_create(const char*,uint32_t,uint32_t,uint32_t),oscar_write(const char*,uint64_t,const void*,size_t,size_t*),oscar_read(const char*,uint64_t,void*,size_t,size_t*),oscar_rename(const char*,const char*,int),oscar_remove(const char*,int),oscar_truncate(const char*,uint64_t);
struct oscar_stat { uint64_t size; uint32_t mode,uid,gid,mtime; };
extern int oscar_stat(const char*,struct oscar_stat*);
struct oscar_info {uint64_t inode,size,atime,mtime,ctime;uint32_t mode,uid,gid,links;};
extern int oscar_open(const char*,int,int*),oscar_close(int),oscar_link(const char*,const char*),oscar_fd_info(int,struct oscar_info*),oscar_fd_read(int,uint64_t,void*,size_t,size_t*),oscar_fd_write(int,uint64_t,const void*,size_t,size_t*,uint64_t*),oscar_fd_times(int,uint32_t,uint64_t,uint64_t);
static void ok(int e){if(e){fprintf(stderr,"adapter errno: %d\n",e);exit(1);}}
int main(int argc,char **argv){
    assert(argc>=3);disk=open(argv[1],O_RDWR);assert(disk>=0);struct stat st;assert(!fstat(disk,&st));
    if(!strcmp(argv[2],"seed") || !strcmp(argv[2],"posix"))ext4_dmask_set(UINT32_MAX);
    int e=oscar_mount(512,st.st_size/512,(!strcmp(argv[2],"readonly") || !strcmp(argv[2],"large") || !strcmp(argv[2],"seed") || !strcmp(argv[2],"integrity")));
    if(!strcmp(argv[2],"integrity")){if(!e){char data[16];size_t n;assert(oscar_read("/many/file-0",0,data,sizeof data,&n)==EIO);}else assert(e==EIO);assert(oscar_error()==EIO);puts("detected metadata checksum failure: EIO");return 0;}
    if(!strcmp(argv[2],"reject")){assert(e);puts("unsupported image rejected");return 0;}ok(e);
    if(!strcmp(argv[2],"seed")){unsigned char b[8192];size_t n;for(int i=0;i<300;i++){char path[64];snprintf(path,sizeof path,"/many/file-%d",i);ok(oscar_read(path,0,b,sizeof b,&n));assert(n==12 && !memcmp(b,"seed fixture",12));}for(int i=0;i<20;i++){ok(oscar_read("/extents",(uint64_t)i*8192,b,4096,&n));assert(n==4096);for(size_t j=0;j<n;j++)assert(b[j]==0);}assert(oscar_create("/denied",0100644,0,0)==EROFS);ok(oscar_unmount());puts("stored metadata checksum seed: read-only ok");return 0;}
    if(!strcmp(argv[2],"readonly")){assert(oscar_create("/denied",0100644,0,0)==EROFS);ok(oscar_unmount());puts("readonly: ok");return 0;}
    if(!strcmp(argv[2],"verify-crash")){struct oscar_stat info;assert(oscar_stat("/gone",&info)==ENOENT);ok(oscar_sync());_exit(77);}
    if(!strcmp(argv[2],"reap-crash") || !strcmp(argv[2],"reap-cut")){
        ok(oscar_create("/gone",0100600,0,0));int fd;ok(oscar_open("/gone",O_RDWR,&fd));size_t n;ok(oscar_write("/gone",0,"orphan",6,&n));ok(oscar_remove("/gone",0));if(argc>3)stop_after=strtol(argv[3],0,10);ok(oscar_close(fd));ok(oscar_sync());_exit(77);
    }
    if(!strcmp(argv[2],"rename-prepare")){
        size_t n;ok(oscar_create("/source",0100644,0,0));ok(oscar_create("/target",0100644,0,0));ok(oscar_write("/source",0,"new",3,&n));ok(oscar_write("/target",0,"old",3,&n));ok(oscar_sync());ok(oscar_unmount());return 0;
    }
    if(!strcmp(argv[2],"rename-cut")){
        int fd;ok(oscar_open("/target",O_RDWR,&fd));stop_after=strtol(argv[3],0,10);ok(oscar_rename("/source","/target",0));_exit(77);
    }
    if(!strcmp(argv[2],"rename-verify")){
        char data[3];size_t n;ok(oscar_read("/target",0,data,3,&n));assert(n==3);
        if(!memcmp(data,"old",3)){ok(oscar_read("/source",0,data,3,&n));assert(n==3 && !memcmp(data,"new",3));}
        else {assert(!memcmp(data,"new",3));struct oscar_stat st;assert(oscar_stat("/source",&st)==ENOENT);}
        ok(oscar_unmount());puts("replacement crash atomicity: ok");return 0;
    }
    if(!strcmp(argv[2],"posix") || !strcmp(argv[2],"orphancrash")){
        ok(oscar_create("/parent",0040700,0,0));ok(oscar_create("/parent/child",0100600,0,0));
        ok(oscar_create("/old",0100644,0,0));size_t n;ok(oscar_write("/old",0,"old",3,&n));int fd;ok(oscar_open("/old",O_RDWR,&fd));
        ok(oscar_link("/old","/alias"));struct oscar_info info;ok(oscar_fd_info(fd,&info));assert(info.links==2);
        uint64_t atime=1700000000123456789ULL,mtime=1700000000987654321ULL;ok(oscar_fd_times(fd,3,atime,mtime));ok(oscar_fd_info(fd,&info));assert(info.atime==atime && info.mtime==mtime);
        ok(oscar_remove("/old",0));ok(oscar_remove("/alias",0));ok(oscar_fd_info(fd,&info));assert(info.links==0);
        uint64_t end;ok(oscar_fd_write(fd,0,"kept",4,&n,&end));char data[8];ok(oscar_fd_read(fd,0,data,sizeof data,&n));assert(n==4 && !memcmp(data,"kept",4));
        if(!strcmp(argv[2],"orphancrash"))_exit(77);
        ok(oscar_create("/old",0100644,0,0));int fresh;ok(oscar_open("/old",O_RDWR,&fresh));struct oscar_info other;ok(oscar_fd_info(fresh,&other));assert(other.inode!=info.inode);ok(oscar_close(fresh));ok(oscar_close(fd));
        ok(oscar_create("/source",0100644,0,0));ok(oscar_write("/source",0,"new",3,&n));ok(oscar_write("/old",0,"target",6,&n));ok(oscar_open("/old",O_RDWR,&fd));ok(oscar_rename("/source","/old",0));ok(oscar_fd_read(fd,0,data,sizeof data,&n));assert(n==6 && !memcmp(data,"target",6));ok(oscar_close(fd));ok(oscar_read("/old",0,data,sizeof data,&n));assert(n==3 && !memcmp(data,"new",3));
        ok(oscar_sync());ok(oscar_unmount());puts("POSIX inode lifetime: ok");return 0;
    }
    if(!strcmp(argv[2],"write") || !strcmp(argv[2],"crash")){
        if(argc>3)stop_after=strtol(argv[3],0,10);
        ok(oscar_create("/persistent",0100644,0,0));
        unsigned char data[60000];for(size_t i=0;i<sizeof data;i++)data[i]=(i*13+7)%251;
        for(unsigned i=0;i<20;i++){size_t n;ok(oscar_write("/persistent",i*sizeof data,data,sizeof data,&n));assert(n==sizeof data);}
        ok(oscar_sync());ok(oscar_rename("/persistent","/renamed",0));
        ok(oscar_create("/removed",0100644,0,0));ok(oscar_remove("/removed",0));
    }else if(!strcmp(argv[2],"verify")){
        unsigned char data[60000];for(unsigned i=0;i<20;i++){size_t n;ok(oscar_read("/renamed",i*sizeof data,data,sizeof data,&n));assert(n==sizeof data);for(size_t j=0;j<n;j++)assert(data[j]==(j*13+7)%251);}
        ok(oscar_truncate("/renamed",17));size_t n;ok(oscar_read("/renamed",0,data,sizeof data,&n));assert(n==17);
    }else if(!strcmp(argv[2],"large")){
        struct oscar_stat info;ok(oscar_stat("/large",&info));assert(info.size==5368709243ULL);
        unsigned char b[123];size_t n;ok(oscar_read("/large",5368709120ULL,b,sizeof b,&n));assert(n==sizeof b);
        for(size_t i=0;i<n;i++)assert(b[i]==0);
    }else if(!strcmp(argv[2],"flusherror")){
        ok(oscar_create("/error",0100644,0,0));flush_failed=1;size_t n;assert(oscar_write("/error",0,"x",1,&n)==EIO);assert(oscar_sync()==EIO);assert(oscar_create("/later",0100644,0,0)==EIO);puts("flush error: propagated");return 0;
    }else if(!strcmp(argv[2],"ioerror")){
        ok(oscar_create("/error",0100644,0,0));failed=1;size_t n;assert(oscar_write("/error",0,"x",1,&n)==EIO);puts("I/O error: propagated");return 0;
    }else if(!strcmp(argv[2],"recover")){ok(oscar_sync());}
    ok(oscar_unmount());close(disk);puts("adapter: ok");return 0;
}
