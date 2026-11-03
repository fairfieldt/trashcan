/* Oscar's single-threaded lwext4 adapter. All device I/O goes through fsd. */
#include "../vendor/lwext4/src/ext4.c"
#include <ext4_super.h>
#include <ext4_inode.h>
#include <ext4_crc32.h>
#include <string.h>
#ifndef EBADF
#define EBADF 9
#endif
#ifndef EMFILE
#define EMFILE 24
#endif
extern int fsd_blocks(int write, uint64_t block, uint32_t count, void *buffer);
extern int fsd_flush(void);
static int io_error;
static int device_readonly;
extern void fsd_integrity(const char *file,unsigned line);
void ext4_metadata_csum_error(const char *file,unsigned line) { if(!io_error)fsd_integrity(file,line);io_error=EIO; }
int oscar_error(void) { return io_error; }
void ext4_port_trans_error(int error) { if(!io_error)io_error=error; }
static int recover_orphans(void);
static int flush_device(void) {int e=fsd_flush();if(e)io_error=e;return e?e:io_error;}
static int dev_open(struct ext4_blockdev *b) { (void)b; return 0; }
static int dev_close(struct ext4_blockdev *b) { (void)b; return flush_device(); }
static int dev_read(struct ext4_blockdev *b, void *buf, uint64_t at, uint32_t n) { (void)b;int e=fsd_blocks(0,at,n,buf);if(e)io_error=e;return e; }
static int dev_write(struct ext4_blockdev *b,const void *buf,uint64_t at,uint32_t n) { (void)b;if(device_readonly)return EROFS;if(io_error)return io_error;int e=fsd_blocks(1,at,n,(void*)buf);if(e)io_error=e;return e?e:flush_device(); }
static unsigned char physical[4096];
static struct ext4_blockdev_iface iface={.open=dev_open,.bread=dev_read,.bwrite=dev_write,.close=dev_close,.ph_bbuf=physical};
static struct ext4_blockdev device={.bdif=&iface};
int oscar_mount(uint32_t block_size,uint64_t blocks,int readonly) {
    if(block_size>sizeof physical || block_size<512 || (block_size&(block_size-1))) return EINVAL;
    io_error=0;device_readonly=readonly;
    unsigned char bytes[4096];int e=fsd_blocks(0,0,sizeof(bytes)/block_size,bytes);if(e)return e;
    struct ext4_sblock *sb=(void*)(bytes+1024);
    if(to_le16(sb->magic)!=0xef53 || to_le32(sb->log_block_size)>2)return ENOTSUP;
    uint32_t compat=to_le32(sb->features_compatible),inc=to_le32(sb->features_incompatible),ro=to_le32(sb->features_read_only);
    /* orphan_file can be ignored only for clean, read-only filesystems without
     * ORPHAN_PRESENT. Writable mounts continue to use the classic orphan chain. */
    if(compat&~(EXT4_FCOM_HAS_JOURNAL|EXT4_FCOM_EXT_ATTR|EXT4_FCOM_RESIZE_INODE|EXT4_FCOM_DIR_INDEX|(readonly?0x1000u:0)))return ENOTSUP;
    if(!readonly && (inc&EXT4_FINCOM_CSUM_SEED))return ENOTSUP;
    if(readonly && (compat&0x1000u) && (!(to_le16(sb->state)&EXT4_SUPERBLOCK_STATE_VALID_FS) || to_le32(sb->last_orphan)))return EROFS;
    if(inc&~(EXT4_SUPPORTED_FINCOM|EXT4_FINCOM_RECOVER))return ENOTSUP;
    if(ro&~EXT4_SUPPORTED_FRO_COM)return ENOTSUP;
    if(readonly && inc&EXT4_FINCOM_RECOVER)return EROFS;
    if(!readonly && !(compat&EXT4_FCOM_HAS_JOURNAL))return ENOTSUP;
    iface.ph_bsize=block_size;iface.ph_bcnt=blocks;device.part_size=blocks*block_size;
    e=ext4_device_register(&device,"disk");if(e)return e;
    e=ext4_mount("disk","/",readonly);if(e)return e;
    if(!readonly) { e=ext4_recover("/");if(e && e!=ENOTSUP)return e; e=ext4_journal_start("/");if(!e)e=recover_orphans(); }
    return e?e:io_error;
}
int oscar_sync(void) { int e=ext4_cache_flush("/");return e?e:flush_device(); }
int oscar_unmount(void) {int e=oscar_sync();if(e)return e;e=ext4_journal_stop("/");if(e)return e;return ext4_umount("/");}
struct oscar_stat { uint64_t size; uint32_t mode,uid,gid,mtime; };
int oscar_stat(const char *path,struct oscar_stat *s) {
    struct ext4_inode inode;int e=ext4_raw_inode_fill(path,0,&inode);if(e)return e;
    memset(s,0,sizeof *s);struct ext4_sblock *sb;e=ext4_get_sblock("/",&sb);if(e)return e;s->size=ext4_inode_get_size(sb,&inode);
    e=ext4_mode_get(path,&s->mode);if(e)return e;
    e=ext4_owner_get(path,&s->uid,&s->gid);if(e)return e;
    e=ext4_mtime_get(path,&s->mtime);return e?e:io_error;
}
int oscar_read(const char *path,uint64_t off,void *buf,size_t len,size_t *done) {
    ext4_file f;int e=ext4_fopen(&f,path,"r");if(e)return e;
    if(off>=ext4_fsize(&f)){*done=0;return ext4_fclose(&f);}
    e=ext4_fseek(&f,off,SEEK_SET);if(!e)e=ext4_fread(&f,buf,len,done);
    int end=ext4_fclose(&f);return e?e:end?end:io_error;
}
int oscar_write(const char *path,uint64_t off,const void *buf,size_t len,size_t *done) {
    ext4_file f;int e=ext4_fopen(&f,path,"r+");if(e)return e;
    if(off>ext4_fsize(&f)) {static const unsigned char zero[4096];uint64_t gap=off-ext4_fsize(&f);e=ext4_fseek(&f,ext4_fsize(&f),SEEK_SET);while(!e && gap){size_t n,w=gap>sizeof zero?sizeof zero:gap;e=ext4_fwrite(&f,zero,w,&n);if(!e && n!=w)e=EIO;gap-=n;}if(e){ext4_fclose(&f);return e;}}
    e=ext4_fseek(&f,off,SEEK_SET);if(!e)e=ext4_fwrite(&f,buf,len,done);
    int end=ext4_fclose(&f);return e?e:end?end:io_error;
}
int oscar_create(const char *path,uint32_t mode,uint32_t uid,uint32_t gid) {
    if(io_error)return io_error;
    int e;if((mode&0170000)==0040000)e=ext4_dir_mk(path);else {ext4_file f;e=ext4_fopen2(&f,path,O_WRONLY|O_CREAT|O_EXCL);if(!e)e=ext4_fclose(&f);}
    if(e)return e;e=ext4_mode_set(path,mode);if(e)return e;e=ext4_owner_set(path,uid,gid);return e?e:io_error;
}
int oscar_truncate(const char *path,uint64_t size) {if(io_error)return io_error;ext4_file f;int e=ext4_fopen(&f,path,"r+");if(e)return e;if(size>ext4_fsize(&f)){ext4_fclose(&f);size_t n;return oscar_write(path,size-1,"\0",1,&n);}e=ext4_ftruncate(&f,size);int end=ext4_fclose(&f);return e?e:end?end:io_error;}
int oscar_remove(const char *path,int directory);
int oscar_rename(const char *path,const char *next,int directory);
int oscar_symlink(const char *path,const char *target) {if(io_error)return io_error;int e=ext4_fsymlink(target,path);return e?e:io_error;}
int oscar_readlink(const char *path,char *buf,size_t len,size_t *done) {int e=ext4_readlink(path,buf,len,done);return e?e:io_error;}
/* Directory iteration stays opaque to Rust. One server processes one request at a time. */
static ext4_dir dir;
int oscar_dir_open(const char *path) {return ext4_dir_open(&dir,path);}
const ext4_direntry *oscar_dir_next(void) {return ext4_dir_entry_next(&dir);}
int oscar_dir_close(void) {int e=ext4_dir_close(&dir);return e?e:io_error;}
int oscar_setattr(const char *path,uint32_t mode,uint32_t uid,uint32_t gid) {
    if(io_error)return io_error;int e=0;
    if(mode!=UINT32_MAX)e=ext4_mode_set(path,mode);
    if(!e && (uid!=UINT32_MAX || gid!=UINT32_MAX)) {
        uint32_t old_uid,old_gid;e=ext4_owner_get(path,&old_uid,&old_gid);
        if(!e)e=ext4_owner_set(path,uid==UINT32_MAX?old_uid:uid,gid==UINT32_MAX?old_gid:gid);
    }return e?e:io_error;
}
int oscar_times(const char *path,uint32_t flags,uint32_t atime,uint32_t mtime) {
    if(io_error)return io_error;int e=0;
    if(flags&1)e=ext4_atime_set(path,atime);
    if(!e && flags&2)e=ext4_mtime_set(path,mtime);
    return e?e:io_error;
}

/* Open identities are private to fsd. Each is owned by a separate channel, never
 * by a caller-supplied inode number. Inodes with no names stay on ext4's classic
 * orphan chain until the final open description closes. */
#define OSCAR_OPEN_MAX 256
static struct { int used; ext4_file file; } opened[OSCAR_OPEN_MAX];
static int pins(uint32_t inode) {int n=0;for(unsigned i=0;i<OSCAR_OPEN_MAX;i++)if(opened[i].used && opened[i].file.inode==inode)n++;return n;}
static int journal_super(struct ext4_mountpoint *mp) { return ext4_sb_write_trans(mp->fs.bdev,&mp->fs.sb); }

static int add_orphan(struct ext4_mountpoint *mp,struct ext4_inode_ref *inode) {
    ext4_inode_set_del_time(inode->inode,ext4_get32(&mp->fs.sb,last_orphan));
    inode->dirty=true;ext4_set32(&mp->fs.sb,last_orphan,inode->index);return journal_super(mp);
}
static int detach_orphan(struct ext4_mountpoint *mp,uint32_t inode,uint32_t next) {
    uint32_t current=ext4_get32(&mp->fs.sb,last_orphan),limit=ext4_get32(&mp->fs.sb,inodes_count);
    if(current==inode){ext4_set32(&mp->fs.sb,last_orphan,next);return journal_super(mp);}
    for(uint32_t step=0;current && step<limit;step++) {
        struct ext4_inode_ref ref;int e=ext4_fs_get_inode_ref(&mp->fs,current,&ref);if(e)return e;
        uint32_t following=ext4_inode_get_del_time(ref.inode);
        if(following==inode){ext4_inode_set_del_time(ref.inode,next);ref.dirty=true;e=ext4_fs_put_inode_ref(&ref);return e;}
        e=ext4_fs_put_inode_ref(&ref);if(e)return e;current=following;
    }return EIO;
}
static int reap_inode(struct ext4_mountpoint *mp,uint32_t inode) {
    if(pins(inode))return 0;
    int e=ext4_trunc_inode(mp,inode,0);if(e)return e;
    e=ext4_trans_start(mp);if(e)return e;
    struct ext4_inode_ref ref;e=ext4_fs_get_inode_ref(&mp->fs,inode,&ref);if(e){ext4_trans_abort(mp);return e;}
    uint32_t next=ext4_inode_get_del_time(ref.inode);
    e=detach_orphan(mp,inode,next);
    if(!e){ext4_inode_set_del_time(ref.inode,UINT32_MAX);ref.dirty=true;e=ext4_fs_free_inode(&ref);}
    int end=ext4_fs_put_inode_ref(&ref);if(!e)e=end;
    if(e)ext4_trans_abort(mp);else e=ext4_trans_stop(mp);
    return e?e:io_error;
}
static int recover_orphans(void) {
    struct ext4_mountpoint *mp=ext4_get_mount("/");
    // Journal replay handles block zero specially; reread the head for 1 KiB blocks too.
    struct ext4_sblock disk_sb;int e=ext4_sb_read(mp->fs.bdev,&disk_sb);if(e)return e;
    mp->fs.sb.last_orphan=disk_sb.last_orphan;
    uint32_t limit=ext4_get32(&mp->fs.sb,inodes_count);
    for(uint32_t steps=0;ext4_get32(&mp->fs.sb,last_orphan);steps++) {
        if(steps>=limit)return EIO;
        uint32_t inode=ext4_get32(&mp->fs.sb,last_orphan);
        struct ext4_inode_ref ref;e=ext4_fs_get_inode_ref(&mp->fs,inode,&ref);if(e)return e;
        int live=ext4_inode_get_links_cnt(ref.inode)!=0;ext4_fs_put_inode_ref(&ref);
        if(live)return EIO;
        e=reap_inode(mp,inode);if(e)return e;
    }return 0;
}
static int remove_in_transaction(struct ext4_mountpoint *mp,const char *path,int directory,uint32_t *orphan) {
    ext4_file f;uint32_t parent_id,name_off;int e=ext4_generic_open2(&f,path,O_RDONLY,EXT4_DE_UNKNOWN,&parent_id,&name_off);if(e)return e;
    struct ext4_inode_ref parent,child;e=ext4_fs_get_inode_ref(&mp->fs,parent_id,&parent);if(e)return e;
    e=ext4_fs_get_inode_ref(&mp->fs,f.inode,&child);if(e){ext4_fs_put_inode_ref(&parent);return e;}
    int isdir=ext4_inode_is_type(&mp->fs.sb,child.inode,EXT4_INODE_MODE_DIRECTORY);
    if(isdir!=directory)e=directory?ENOTDIR:EISDIR;
    bool goal;int len=ext4_path_check(path+name_off,&goal);
    if(!e)e=ext4_unlink(mp,&parent,&child,path+name_off,len);
    if(!e && isdir){ext4_inode_set_links_cnt(child.inode,0);child.dirty=true;}
    if(!e && !ext4_inode_get_links_cnt(child.inode)){e=add_orphan(mp,&child);if(!e)*orphan=child.index;}
    int end=ext4_fs_put_inode_ref(&child);if(!e)e=end;end=ext4_fs_put_inode_ref(&parent);if(!e)e=end;return e;
}
int oscar_remove(const char *path,int directory) {
    if(!strcmp(path,"/"))return 16;
    struct ext4_mountpoint *mp=ext4_get_mount(path);if(!mp)return ENOENT;if(mp->fs.read_only)return EROFS;if(io_error)return io_error;
    uint32_t orphan=0,old_head=mp->fs.sb.last_orphan;int e=ext4_trans_start(mp);if(e)return e;
    e=remove_in_transaction(mp,path,directory,&orphan);
    if(e){ext4_trans_abort(mp);mp->fs.sb.last_orphan=old_head;}else e=ext4_trans_stop(mp);
    if(!e && orphan)e=reap_inode(mp,orphan);return e?e:io_error;
}
int oscar_rename(const char *path,const char *next,int directory) {
    if(!strcmp(path,"/") || !strcmp(next,"/"))return 16;
    (void)directory;struct ext4_mountpoint *mp=ext4_get_mount(path);if(!mp)return ENOENT;if(mp->fs.read_only)return EROFS;if(io_error)return io_error;
    ext4_file source,target;uint32_t parent_id,name_off;
    int e=ext4_generic_open2(&source,path,O_RDONLY,EXT4_DE_UNKNOWN,&parent_id,&name_off);if(e)return e;
    int found=ext4_generic_open2(&target,next,O_RDONLY,EXT4_DE_UNKNOWN,NULL,NULL);
    if(found && found!=ENOENT)return found;if(!found && target.inode==source.inode)return 0;
    struct ext4_inode_ref parent,child;e=ext4_fs_get_inode_ref(&mp->fs,parent_id,&parent);if(e)return e;
    e=ext4_fs_get_inode_ref(&mp->fs,source.inode,&child);if(e){ext4_fs_put_inode_ref(&parent);return e;}
    int isdir=ext4_inode_is_type(&mp->fs.sb,child.inode,EXT4_INODE_MODE_DIRECTORY);
    uint32_t orphan=0,old_head=mp->fs.sb.last_orphan;e=ext4_trans_start(mp);
    if(!e && !found)e=remove_in_transaction(mp,next,isdir,&orphan);
    if(!e)e=ext4_create_hardlink(next,&child,true);
    if(!e)e=ext4_remove_orig_reference(path,name_off,&parent,&child);
    int end=ext4_fs_put_inode_ref(&child);if(!e)e=end;end=ext4_fs_put_inode_ref(&parent);if(!e)e=end;
    if(e){ext4_trans_abort(mp);mp->fs.sb.last_orphan=old_head;}else e=ext4_trans_stop(mp);
    if(!e && orphan)e=reap_inode(mp,orphan);return e?e:io_error;
}
int oscar_link(const char *path,const char *next){if(io_error)return io_error;struct ext4_inode inode;int e=ext4_raw_inode_fill(path,NULL,&inode);if(e)return e;if(ext4_inode_get_links_cnt(&inode)==UINT16_MAX)return 31;e=ext4_flink(path,next);return e?e:io_error;}

struct oscar_info {uint64_t inode,size,atime,mtime,ctime;uint32_t mode,uid,gid,links;};
static uint64_t decode_time(struct ext4_sblock *sb,struct ext4_inode *inode,uint32_t low,uint32_t extra,unsigned offset) {
    uint64_t seconds=to_le32(low),ns=0;
    if(ext4_get16(sb,inode_size)>=offset+4 && 128+ext4_inode_get_extra_isize(sb,inode)>=offset+4){uint32_t value=to_le32(extra);seconds|=(uint64_t)(value&3)<<32;ns=value>>2;}
    return seconds*1000000000ULL+ns;
}
static int inode_info(struct ext4_mountpoint *mp,uint32_t inode,struct oscar_info *info) {
    struct ext4_inode_ref ref;int e=ext4_fs_get_inode_ref(&mp->fs,inode,&ref);if(e)return e;
    struct ext4_inode *i=ref.inode;struct ext4_sblock *sb=&mp->fs.sb;
    *info=(struct oscar_info){.inode=inode,.size=ext4_inode_get_size(sb,i),.mode=ext4_inode_get_mode(sb,i),.uid=ext4_inode_get_uid(i),.gid=ext4_inode_get_gid(i),.links=ext4_inode_get_links_cnt(i),
        .atime=decode_time(sb,i,i->access_time,i->atime_extra,offsetof(struct ext4_inode,atime_extra)),
        .mtime=decode_time(sb,i,i->modification_time,i->mtime_extra,offsetof(struct ext4_inode,mtime_extra)),
        .ctime=decode_time(sb,i,i->change_inode_time,i->ctime_extra,offsetof(struct ext4_inode,ctime_extra))};
    return ext4_fs_put_inode_ref(&ref);
}
int oscar_info(const char *path,struct oscar_info *info){ext4_file f;int e=ext4_generic_open2(&f,path,O_RDONLY,EXT4_DE_UNKNOWN,NULL,NULL);return e?e:inode_info(f.mp,f.inode,info);}
int oscar_open(const char *path,int flags,int *fd){for(unsigned i=0;i<OSCAR_OPEN_MAX;i++)if(!opened[i].used){int e=ext4_generic_open2(&opened[i].file,path,flags&3,EXT4_DE_UNKNOWN,NULL,NULL);if(e)return e;opened[i].used=1;*fd=i+1;return 0;}return EMFILE;}
static ext4_file *file_of(int fd){if(fd<1 || fd>OSCAR_OPEN_MAX || !opened[fd-1].used)return NULL;return &opened[fd-1].file;}
int oscar_close(int fd){ext4_file *f=file_of(fd);if(!f)return EBADF;struct ext4_mountpoint *mp=f->mp;uint32_t inode=f->inode;opened[fd-1].used=0;ext4_fclose(f);struct oscar_info info;int e=inode_info(mp,inode,&info);if(!e && !info.links)e=reap_inode(mp,inode);return e?e:io_error;}
int oscar_fd_info(int fd,struct oscar_info *info){ext4_file *f=file_of(fd);return f?inode_info(f->mp,f->inode,info):EBADF;}
int oscar_fd_read(int fd,uint64_t off,void *data,size_t len,size_t *done){ext4_file *f=file_of(fd);if(!f)return EBADF;if((f->flags&3)==O_WRONLY)return EBADF;struct oscar_info i;int e=inode_info(f->mp,f->inode,&i);if(e)return e;f->fsize=i.size;if(off>=i.size){*done=0;return 0;}e=ext4_fseek(f,off,SEEK_SET);if(!e)e=ext4_fread(f,data,len,done);return e?e:io_error;}
int oscar_fd_write(int fd,uint64_t off,const void *data,size_t len,size_t *done,uint64_t *end){ext4_file *f=file_of(fd);if(!f)return EBADF;if((f->flags&3)==O_RDONLY)return EBADF;if(io_error)return io_error;struct oscar_info i;int e=inode_info(f->mp,f->inode,&i);if(e)return e;f->fsize=i.size;if(off==UINT64_MAX)off=i.size;
    if(off>i.size){static const char zero[4096];e=ext4_fseek(f,i.size,SEEK_SET);uint64_t gap=off-i.size;while(!e && gap){size_t n=0,w=gap>sizeof zero?sizeof zero:gap;e=ext4_fwrite(f,zero,w,&n);if(!e && n!=w)e=EIO;gap-=n;}}
    if(!e)e=ext4_fseek(f,off,SEEK_SET);if(!e)e=ext4_fwrite(f,data,len,done);*end=off+*done;return e?e:io_error;}
int oscar_fd_truncate(int fd,uint64_t size){ext4_file *f=file_of(fd);if(!f)return EBADF;if((f->flags&3)==O_RDONLY)return EBADF;struct oscar_info i;int e=inode_info(f->mp,f->inode,&i);if(e)return e;f->fsize=i.size;if(size>i.size){size_t n;uint64_t end;return oscar_fd_write(fd,size-1,"\0",1,&n,&end);}e=ext4_ftruncate(f,size);return e?e:io_error;}
int oscar_fd_setattr(int fd,uint32_t mode,uint32_t uid,uint32_t gid){ext4_file *f=file_of(fd);if(!f)return EBADF;if(f->mp->fs.read_only)return EROFS;if(io_error)return io_error;int e=ext4_trans_start(f->mp);if(e)return e;struct ext4_inode_ref ref;e=ext4_fs_get_inode_ref(&f->mp->fs,f->inode,&ref);if(!e){if(mode!=UINT32_MAX)ext4_inode_set_mode(&f->mp->fs.sb,ref.inode,mode);if(uid!=UINT32_MAX)ext4_inode_set_uid(ref.inode,uid);if(gid!=UINT32_MAX)ext4_inode_set_gid(ref.inode,gid);ref.dirty=true;e=ext4_fs_put_inode_ref(&ref);}if(e)ext4_trans_abort(f->mp);else e=ext4_trans_stop(f->mp);return e?e:io_error;}
int oscar_fd_times(int fd,uint32_t flags,uint64_t atime,uint64_t mtime){ext4_file *f=file_of(fd);if(!f)return EBADF;if(f->mp->fs.read_only)return EROFS;if(io_error)return io_error;int e=ext4_trans_start(f->mp);if(e)return e;struct ext4_inode_ref ref;e=ext4_fs_get_inode_ref(&f->mp->fs,f->inode,&ref);if(!e){struct ext4_inode *i=ref.inode;
    uint64_t times[2]={atime,mtime};unsigned offsets[2]={offsetof(struct ext4_inode,atime_extra),offsetof(struct ext4_inode,mtime_extra)};
    for(unsigned n=0;n<2 && !e;n++)if(flags&(1u<<n)){uint64_t seconds=times[n]/1000000000ULL;uint32_t ns=times[n]%1000000000ULL;int extra=ext4_get16(&f->mp->fs.sb,inode_size)>=offsets[n]+4 && 128+ext4_inode_get_extra_isize(&f->mp->fs.sb,i)>=offsets[n]+4;if(seconds>>34 || (!extra && (ns || seconds>>32)))e=ENOTSUP;}
    if(!e)for(unsigned n=0;n<2;n++)if(flags&(1u<<n)){uint64_t seconds=times[n]/1000000000ULL;uint32_t ns=times[n]%1000000000ULL;int extra=ext4_get16(&f->mp->fs.sb,inode_size)>=offsets[n]+4 && 128+ext4_inode_get_extra_isize(&f->mp->fs.sb,i)>=offsets[n]+4;uint32_t value=to_le32(ns<<2 | (seconds>>32));if(n==0){i->access_time=to_le32(seconds);if(extra)i->atime_extra=value;}else{i->modification_time=to_le32(seconds);if(extra)i->mtime_extra=value;}ref.dirty=true;}
    int end=ext4_fs_put_inode_ref(&ref);if(!e)e=end;}if(e)ext4_trans_abort(f->mp);else e=ext4_trans_stop(f->mp);return e?e:io_error;}
