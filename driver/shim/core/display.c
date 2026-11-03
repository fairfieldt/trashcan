// SPDX-License-Identifier: MIT
/* Current-mode SI display handover. Register offsets, vblank counter and flip
 * sequence are from Linux 7.3-rc7 dce_v6_0.c (AMD 2015, MIT), retained unmodified
 * under driver/linux. Connector training/timings remain firmware-owned. */
#include "amdgpu.h"
#include "sid.h"
#include "dce/dce_6_0_d.h"
#include "dce/dce_6_0_sh_mask.h"
#include "gmc/gmc_6_0_d.h"
#include <gpud_core.h>

static const u32 offsets[6] = { CRTC0_REGISTER_OFFSET, CRTC1_REGISTER_OFFSET,
    CRTC2_REGISTER_OFFSET, CRTC3_REGISTER_OFFSET, CRTC4_REGISTER_OFFSET, CRTC5_REGISTER_OFFSET };
static struct {
    struct gpud_display_info info;
    u64 secondary;
    u32 flip_control;
    u64 buffers[2];
    struct amdgpu_bo *staging;
    struct amdgpu_ib ib;
    struct amdgpu_bo *ib_bo;
    bool owned;
} display[6];
static u64 firmware_vram_base;

void gpud_display_capture(struct amdgpu_device *adev)
{
    memset(display,0,sizeof(display));
    firmware_vram_base=(u64)(RREG32(mmMC_VM_FB_LOCATION)&0xffff)<<24;
    for (unsigned i=0;i<6;i++) {
        u32 off=offsets[i], crtc=RREG32(mmCRTC_CONTROL+off), enabled=RREG32(mmGRPH_ENABLE+off);
        struct gpud_display_info *d=&display[i].info;
        if (!(crtc&CRTC_CONTROL__CRTC_MASTER_EN_MASK) || !(enabled&1)) continue;
        u32 viewport=RREG32(mmVIEWPORT_SIZE+off);
        d->index=i;d->width=viewport>>16;d->height=viewport&0xffff;
        d->pitch=RREG32(mmGRPH_PITCH+off);
        d->control=RREG32(mmGRPH_CONTROL+off);d->swap=RREG32(mmGRPH_SWAP_CNTL+off);
        d->base=(u64)RREG32(mmGRPH_PRIMARY_SURFACE_ADDRESS_HIGH+off)<<32;
        d->base|=RREG32(mmGRPH_PRIMARY_SURFACE_ADDRESS+off)&GRPH_PRIMARY_SURFACE_ADDRESS__GRPH_PRIMARY_SURFACE_ADDRESS_MASK;
        display[i].secondary=(u64)RREG32(mmGRPH_SECONDARY_SURFACE_ADDRESS_HIGH+off)<<32;
        display[i].secondary|=RREG32(mmGRPH_SECONDARY_SURFACE_ADDRESS+off)&0xffffff00;
        display[i].flip_control=RREG32(mmGRPH_FLIP_CONTROL+off);
        d->bytes=(u64)d->pitch*d->height*4;d->flags=1;
        unsigned array=(d->control&GRPH_CONTROL__GRPH_ARRAY_MODE_MASK)>>GRPH_CONTROL__GRPH_ARRAY_MODE__SHIFT;
        if ((d->control&GRPH_CONTROL__GRPH_DEPTH_MASK)==GRPH_DEPTH_32BPP &&
            !(d->control&GRPH_CONTROL__GRPH_FORMAT_MASK) && array<=GRPH_ARRAY_LINEAR_ALIGNED &&
            d->width && d->height && d->width<=4096 && d->height<=4096 && d->pitch>=d->width &&
            d->pitch<=8192 && d->bytes<=(64ULL<<20)) d->flags|=2;
        dev_info(adev->dev,"firmware display %u: %ux%u pitch %u, MC %#llx size %llu, control %#x swap %#x\n",
            i,d->width,d->height,d->pitch,(unsigned long long)d->base,(unsigned long long)d->bytes,d->control,d->swap);
    }
}
int gpud_display_reservation(unsigned which,u64 *offset,u64 *size)
{
    if (which>=12) return -EINVAL;
    struct gpud_display_info *d=&display[which/2].info;
    u64 base=(which&1)?display[which/2].secondary:d->base;
    if (!(d->flags&1) || !d->bytes || base<firmware_vram_base) return -ENOENT;
    *offset=base-firmware_vram_base;*size=ALIGN(d->bytes,PAGE_SIZE);return 0;
}
void gpud_display_pm(struct amdgpu_device *adev)
{
    struct amd_pp_display_configuration *cfg=&adev->pm.pm_display_cfg;
    cfg->num_display=0;cfg->min_vblank_time=0xffffffff;
    for(unsigned i=0;i<6;i++) if(display[i].info.flags&1) {
        if(!cfg->num_display)cfg->crtc_index=i;
        cfg->num_display++;
        /* Without recomputing firmware watermarks/timing, SI's documented
         * zero-vblank path disables MCLK switching while a display is live. */
        cfg->min_vblank_time=0;
    }
}
int gpud_display_info(struct amdgpu_device *adev,unsigned index,struct gpud_display_info *out)
{
    if(index>=6 || !(display[index].info.flags&1)) return -ENOENT;
    *out=display[index].info;out->frame=RREG32(mmCRTC_STATUS_FRAME_COUNT+offsets[index]);return 0;
}


int gpud_display_acquire(struct amdgpu_device *adev,unsigned index,void **cpu,u64 *bytes)
{
    if(index>=6 || (display[index].info.flags&3)!=3 || display[index].owned) return -EINVAL;
    if(adev->gmc.vram_start!=firmware_vram_base || !adev->mman.buffer_funcs) return -EINVAL;
    u64 size=ALIGN(display[index].info.bytes,PAGE_SIZE),dummy;
    int r=gpud_vram_alloc(adev,size,PAGE_SIZE,&display[index].buffers[0]);
    if(r)return r;
    r=gpud_vram_alloc(adev,size,PAGE_SIZE,&display[index].buffers[1]);
    if(r){gpud_vram_free(adev,display[index].buffers[0],size);return r;}
    r=amdgpu_bo_create_kernel(adev,size,PAGE_SIZE,AMDGPU_GEM_DOMAIN_GTT,&display[index].staging,&dummy,cpu);
    if(!r)r=amdgpu_bo_create_kernel(adev,4096,PAGE_SIZE,AMDGPU_GEM_DOMAIN_VRAM,&display[index].ib_bo,&display[index].ib.gpu_addr,(void**)&display[index].ib.ptr);
    if(r) {
        gpud_bo_retire_kernel(adev,&display[index].staging);
        gpud_vram_free(adev,display[index].buffers[0],size);gpud_vram_free(adev,display[index].buffers[1],size);return r;
    }
    *bytes=size;display[index].owned=true;return 0;
}
int gpud_display_upload(struct amdgpu_device *adev,unsigned index,unsigned buffer,int *ring,u64 *sequence)
{
    if(index>=6 || buffer>1 || !display[index].owned)return -EINVAL;
    const struct amdgpu_buffer_funcs *copy=adev->mman.buffer_funcs;
    struct amdgpu_ib *ib=&display[index].ib;
    u64 size=display[index].info.bytes,done=0;
    ib->length_dw=0;
    while(done<size) {
        u32 n=min_t(u64,copy->copy_max_bytes,size-done);
        if(ib->length_dw+copy->copy_num_dw+8>1024)return -E2BIG;
        copy->emit_copy_buffer(ib,display[index].staging->gpu_addr+done,
            adev->gmc.vram_start+display[index].buffers[buffer]+done,n,0);
        done+=n;
    }
    *ring=gpud_ring_for(adev,AMDGPU_HW_IP_DMA,0);
    if(*ring<0)return *ring;
    amdgpu_ring_pad_ib(adev->rings[*ring],ib);
    struct gpud_ib submit={.va=ib->gpu_addr,.size_dw=ib->length_dw,.flags=AMDGPU_IB_FLAG_EMIT_MEM_SYNC};
    return gpud_submit(adev,*ring,NULL,&submit,1,sequence);
}
static int flip(struct amdgpu_device *adev,unsigned index,u64 base,u32 *frame)
{
    u32 off=offsets[index],before=RREG32(mmCRTC_STATUS_FRAME_COUNT+off);
    /* Linux dce_v6_0_page_flip: vsync, unchanged pixel pitch, high then low
     * address (the low write triggers the double-buffered update), posted read. */
    /* DCE6 pageflip IRQ completion is GRPH_PFLIP_INT_OCCURRED. The INUSE
     * aliases read zero on the tested Tahiti and are not used by Linux DCE6.
     * results/dross/hardware-20261103/{gpu3,v6-auth}-netlog.txt */
    WREG32(mmGRPH_INTERRUPT_STATUS+off,GRPH_INTERRUPT_STATUS__GRPH_PFLIP_INT_CLEAR_MASK);
    RREG32(mmGRPH_INTERRUPT_STATUS+off);
    WREG32(mmGRPH_FLIP_CONTROL+off,0);
    WREG32(mmGRPH_PITCH+off,display[index].info.pitch);
    WREG32(mmGRPH_PRIMARY_SURFACE_ADDRESS_HIGH+off,upper_32_bits(base));
    WREG32(mmGRPH_PRIMARY_SURFACE_ADDRESS+off,lower_32_bits(base));
    RREG32(mmGRPH_PRIMARY_SURFACE_ADDRESS+off);
    u64 deadline=lkpi_ktime_ns()+100*NSEC_PER_MSEC;
    while(lkpi_ktime_ns()<deadline) {
        u64 programmed=(u64)RREG32(mmGRPH_PRIMARY_SURFACE_ADDRESS_HIGH+off)<<32;
        programmed|=RREG32(mmGRPH_PRIMARY_SURFACE_ADDRESS+off)&0xffffff00;
        u32 interrupt=RREG32(mmGRPH_INTERRUPT_STATUS+off);
        u32 current=RREG32(mmCRTC_STATUS_FRAME_COUNT+off);
        if(programmed==base && (interrupt&GRPH_INTERRUPT_STATUS__GRPH_PFLIP_INT_OCCURRED_MASK) && !(RREG32(mmGRPH_UPDATE+off)&GRPH_UPDATE__GRPH_SURFACE_UPDATE_PENDING_MASK) && current!=before) {
            *frame=current;return 0;
        }
        udelay(50);
    }
    dev_err(adev->dev,"display %u flip timeout: wanted %#llx, primary %#x:%08x, frame %u -> %u, update %#x, interrupt %#x/control %#x\n",
        index,(unsigned long long)base,RREG32(mmGRPH_PRIMARY_SURFACE_ADDRESS_HIGH+off),
        RREG32(mmGRPH_PRIMARY_SURFACE_ADDRESS+off),before,
        RREG32(mmCRTC_STATUS_FRAME_COUNT+off),RREG32(mmGRPH_UPDATE+off),
        RREG32(mmGRPH_INTERRUPT_STATUS+off),RREG32(mmGRPH_INTERRUPT_CONTROL+off));
    return -ETIMEDOUT;
}
int gpud_display_flip(struct amdgpu_device *adev,unsigned index,unsigned buffer,u32 *frame)
{
    if(index>=6 || buffer>1 || !display[index].owned)return -EINVAL;
    return flip(adev,index,adev->gmc.vram_start+display[index].buffers[buffer],frame);
}
int gpud_display_release(struct amdgpu_device *adev,unsigned index)
{
    if(index>=6 || !display[index].owned)return -EINVAL;
    u32 frame;
    int r=flip(adev,index,display[index].info.base,&frame);
    if(r)return r; /* Scanout may still use our buffers: quarantine them. */
    WREG32(mmGRPH_FLIP_CONTROL+offsets[index],display[index].flip_control);
    u64 size=ALIGN(display[index].info.bytes,PAGE_SIZE);
    gpud_vram_free(adev,display[index].buffers[0],size);gpud_vram_free(adev,display[index].buffers[1],size);
    gpud_bo_retire_kernel(adev,&display[index].staging);
    gpud_bo_retire_kernel(adev,&display[index].ib_bo);
    display[index].owned=false;return 0;
}
