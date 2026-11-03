#define _GNU_SOURCE
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <spawn.h>
#include <sys/mman.h>
extern char **environ;
static int failures;
static void check(const char *name, int ok) { printf("unix: %s %s (errno %d)\n", ok ? "ok" : "FAIL", name, errno); if(!ok) failures++; }
static int run(const char *command) {
 pid_t p=fork(); if(p<0)return -1; if(!p){execl("/bin/sh","sh","-c",command,(char*)0);_exit(127);} int status; if(waitpid(p,&status,0)!=p)return -1; return WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
}
static volatile sig_atomic_t got_signal;
static void handler(int n) { got_signal=n; }
static void network_checks(void) {
 struct sockaddr_in address = {.sin_family=AF_INET, .sin_port=htons(2345), .sin_addr={.s_addr=htonl(INADDR_ANY)}};
 int first=socket(AF_INET,SOCK_STREAM,0),second=socket(AF_INET,SOCK_STREAM,0);
 int ok=first>=0&&second>=0&&bind(first,(struct sockaddr*)&address,sizeof(address))==0;
 ok&=bind(second,(struct sockaddr*)&address,sizeof(address))==-1&&errno==EADDRINUSE;
 check("exclusive TCP binds",ok);close(first);close(second);
 second=socket(AF_INET,SOCK_STREAM,0);address.sin_port=htons(0);
 socklen_t length=sizeof(address);ok=bind(second,(struct sockaddr*)&address,length)==0&&getsockname(second,(struct sockaddr*)&address,&length)==0&&ntohs(address.sin_port)>=49152;
 check("ephemeral TCP bind",ok);close(second);
 pid_t child=fork();if(!child){setgid(1000);setuid(1000);int fd=socket(AF_INET,SOCK_STREAM,0);address.sin_port=htons(22);_exit(fd>=0&&bind(fd,(struct sockaddr*)&address,sizeof(address))==-1&&errno==EACCES?0:1);}int status=0;waitpid(child,&status,0);
 check("privileged port permission",WIFEXITED(status)&&WEXITSTATUS(status)==0);
}
static void namespace_checks(void) {
 char path[64];int count;
 for(count=0;count<4096;count++){snprintf(path,sizeof(path),"/tmp/entry-%d",count);int fd=open(path,O_CREAT|O_WRONLY,0600);if(fd<0)break;close(fd);}
 int fd=open("/dev/ptmx",O_RDWR|O_NOCTTY);check("PTY namespace exhaustion returns an error",count<4096&&fd==-1&&errno==ENOSPC);if(fd>=0)close(fd);
 for(int i=0;i<count;i++){snprintf(path,sizeof(path),"/tmp/entry-%d",i);unlink(path);}
 fd=posix_openpt(O_RDWR|O_NOCTTY);check("PTY allocation recovers after namespace exhaustion",fd>=0);if(fd>=0)close(fd);
}
int main(int argc,char **argv) {
 if(argc==3&&!strcmp(argv[1],"--check-fd"))return fcntl(atoi(argv[2]),F_GETFD)==-1&&errno==EBADF?0:1;
 setbuf(stdout,0); puts("unix: probe started");
 pid_t spawned;char *spawn_args[]={"sh","-c","exit 9",0};int spawn_status=0;
 int spawn_error=posix_spawn(&spawned,"/bin/sh",0,0,spawn_args,environ);if(!spawn_error)waitpid(spawned,&spawn_status,0);
 check("posix_spawn and vfork",spawn_error==0&&WIFEXITED(spawn_status)&&WEXITSTATUS(spawn_status)==9);
 check("posix_spawn exec error",posix_spawn(&spawned,"/does-not-exist",0,0,spawn_args,environ)==ENOENT);
 int value=7;pid_t child=fork();if(!child){value=99;_exit(23);}int status=0;
 check("fork and wait",child>0&&waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==23&&value==7);
 check("busybox exec",run("/bin/busybox echo applet-exec")==0);
 check("pipe and redirection",run("printf 'hello\\n' | tr a-z A-Z > /tmp/result; test \"$(cat /tmp/result)\" = HELLO")==0);
 check("cwd and writable files",run("mkdir /tmp/work && cd /tmp/work && echo data > a && cp a b && mv b c && test \"$(pwd)\" = /tmp/work && rm a c && cd / && rmdir /tmp/work")==0);
 check("symlinks",run("ln -s /tmp/result /tmp/link && test \"$(cat /tmp/link)\" = HELLO && test \"$(readlink /tmp/link)\" = /tmp/result && rm /tmp/link")==0);
 check("script interpreter",run("printf '#!/bin/sh\\necho script-ok > /tmp/script-result\\n' > /tmp/script; chmod 755 /tmp/script; /tmp/script; test \"$(cat /tmp/script-result)\" = script-ok")==0);
 signal(SIGUSR1,handler); kill(getpid(),SIGUSR1); check("signal handler and return",got_signal==SIGUSR1);
 int master=posix_openpt(O_RDWR|O_NOCTTY);check("open ptmx",master>=0);
 if(master>=0){check("unlock pty",grantpt(master)==0&&unlockpt(master)==0);char *name=ptsname(master);int slave=name?open(name,O_RDWR|O_NOCTTY):-1;check("open slave",slave>=0);
  if(slave>=0){char data[32]={0};write(master,"ab\177c\n",5);int n=read(slave,data,sizeof(data));check("pty canonical erase",n==3&&!memcmp(data,"ac\n",3));
   struct termios attrs;int ok=tcgetattr(slave,&attrs)==0;cfmakeraw(&attrs);ok&=tcsetattr(slave,TCSANOW,&attrs)==0;write(master,"z",1);n=read(slave,data,1);check("pty raw mode",ok&&n==1&&data[0]=='z');
   struct winsize size={40,100,0,0},got={0};check("pty resize",ioctl(master,TIOCSWINSZ,&size)==0&&ioctl(slave,TIOCGWINSZ,&got)==0&&got.ws_row==40&&got.ws_col==100);close(slave);}
  close(master);}
 int fd=open("/tmp/fd-test",O_CREAT|O_RDWR|O_CLOEXEC,0600);char fdtext[24];snprintf(fdtext,sizeof(fdtext),"%d",fd);
 child=fork();if(!child){execl("/bin/unix-probe","unix-probe","--check-fd",fdtext,(char*)0);_exit(127);}waitpid(child,&status,0);check("fd close-on-exec",WIFEXITED(status)&&WEXITSTATUS(status)==0);
 check("dup2 preserves cloexec on same fd",dup2(fd,fd)==fd&&(fcntl(fd,F_GETFD)&FD_CLOEXEC));
 check("large fd rejected",dup2(fd,1000000)==-1&&errno==EBADF);
 write(fd,"A",1);child=fork();if(!child){write(fd,"B",1);_exit(0);}waitpid(child,&status,0);write(fd,"C",1);lseek(fd,0,SEEK_SET);char shared_offset[4]={0};read(fd,shared_offset,3);check("fork shares open file descriptions",!strcmp(shared_offset,"ABC"));
 unlink("/tmp/fd-test");lseek(fd,0,SEEK_SET);check("unlinked fd remains readable",read(fd,shared_offset,3)==3);close(fd);
 void *shared=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);*(volatile int*)shared=3;child=fork();if(!child){*(volatile int*)shared=11;_exit(0);}waitpid(child,&status,0);check("fork preserves shared mappings",*(volatile int*)shared==11);munmap(shared,4096);
 int reclaim=1;for(int i=0;i<20;i++){fd=open("/tmp/quota-test",O_CREAT|O_RDWR|O_TRUNC,0600);if(fd<0||ftruncate(fd,2*1024*1024)<0)reclaim=0;unlink("/tmp/quota-test");close(fd);}check("RAM quota reclaimed after unlink/close",reclaim);
 check("quoted empty arguments",run("test \"$(/bin/busybox printf '<%s>' '')\" = '<>'")==0);
 check("symlink directory creation",run("mkdir /tmp/actual && ln -s /tmp/actual /tmp/alias && echo through-link > /tmp/alias/new && test \"$(cat /tmp/actual/new)\" = through-link && rm /tmp/actual/new /tmp/alias && rmdir /tmp/actual")==0);
 fd=open("/tmp/root-secret",O_CREAT|O_WRONLY,0600);write(fd,"secret",6);close(fd);mkdir("/tmp/private-dir",0700);fd=open("/tmp/private-dir/public-file",O_CREAT|O_WRONLY,0644);close(fd);
 child=fork();if(!child){int ok=setgid(1000)==0&&setuid(1000)==0&&getuid()==1000&&geteuid()==1000;ok&=setuid(0)==-1&&errno==EPERM;ok&=open("/tmp/root-secret",O_RDONLY)==-1&&errno==EACCES;ok&=open("/tmp/private-dir/public-file",O_RDONLY)==-1&&errno==EACCES;_exit(ok?0:1);}waitpid(child,&status,0);check("credential drop and directory/file permissions",WIFEXITED(status)&&WEXITSTATUS(status)==0);

 check("process groups reject a nonexistent group",setpgid(0,2147483647)==-1&&errno==EPERM);
 child=fork();if(!child){signal(SIGCONT,SIG_IGN);got_signal=0;raise(SIGSTOP);_exit(got_signal==SIGUSR1?42:1);}
 int stopped=waitpid(child,&status,WUNTRACED)==child&&WIFSTOPPED(status);kill(child,SIGUSR1);usleep(10000);int stayed=waitpid(child,&status,WNOHANG)==0;kill(child,SIGCONT);waitpid(child,&status,0);
 check("stopped signals wait for ignored SIGCONT",stopped&&stayed&&WIFEXITED(status)&&WEXITSTATUS(status)==42);
 network_checks();namespace_checks();
 check("missing executable",run("/does-not-exist")==127);
 check("host key provisioning",run("/usr/sbin/dropbearconvert openssh dropbear /etc/test-hostkey /etc/dropbear/hostkey")==0);
 pid_t server=fork();if(!server){execl("/usr/sbin/dropbear","dropbear","-F","-r","/etc/dropbear/hostkey","-p","22",(char*)0);_exit(127);}
 check("sshd process",server>0);puts("unix: SSH READY");
 int seen=0;for(int i=0;i<6000;i++){if(access("/tmp/ssh-done",F_OK)==0){seen=1;break;}usleep(10000);}
 check("OpenSSH remote acceptance",seen);kill(server,SIGTERM);while(waitpid(server,&status,0)<0&&errno==EINTR){}
 printf("unix: %s (%d failures)\n",failures?"FAILED":"ALL PASSED",failures);return failures?1:0;
}
