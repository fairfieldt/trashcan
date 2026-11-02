# Imported Linux files

Unmodified copies from Linux 7.3-rc7 (commit 24a19b953e2a, "Linux 7.3-rc7"; kernel.org cgit snapshot
`linux-7.3-rc7.tar.gz`, sha256 f789fc0d7b3713b6be23284f2a9c35bf41fb86785b24cfcc506c20d8ff712523), at their
upstream paths. Every file keeps its license text: AMD's MIT-style permission notice, or for the two DRM uapi headers
`GPL-2.0 WITH Linux-syscall-note` / MIT as marked. Local changes: none. gpud's replacements for what these files
call (amdgpu.h, the core, the kernel API) are in `driver/shim/`.

| File | sha256 |
|---|---|
| `drivers/gpu/drm/amd/amdgpu/amdgpu_gart.h` | a85ed422b1cc0eabde14ef930a41e7c4970fc520e7a0f9f0ee2e7f5990442919 |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_ih.c` | 12214cfc8de6c600cabe49f79778a4a2ecd232a2aaf861da3622ed80b8deb97b |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_ih.h` | a40a0310c00c69adaf0c845c2917294c02cea65aafc7f3e51f66680593d5a3fc |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_ip.h` | dde3a58612933f890921698b3dd05ec8cb4595f4ceeb366baf94ec3f178d7e97 |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_irq.h` | 72aa3eadfe439463982331d5ccf011339d80c91a2556212474e12dfdb04fdc0b |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_reg_access.h` | e47ffc00971107c3ed9aaf06e589dcabb826daaacc14596431e8b5f744476504 |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_ring.h` | d791cc949412dff2ecc906bca12467a42c2d98681a2089fdb569a45b7f27832e |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_rlc.h` | 144e04a6382c18c1690babbf3fb5983fb785c8e65a10701b05dedaa64f64acef |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_socbb.h` | 13b4070f2b0b2ece3d2bd132d67bc107cd64c765c5980d38266d87eb5f22185f |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_ucode.h` | 222b508dfd40897ead5aecd207ab54c015c311c50e96246a962c1eb29c63b1e0 |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_video_codecs.h` | afa67007f660ea6f75d419457d97c393628e4134c17738fd926f98e7590a2070 |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_wb.c` | 40fa170ddd174c021f634f1648011e2f686043db23cd30241de13fd3f0489ec9 |
| `drivers/gpu/drm/amd/amdgpu/amdgpu_wb.h` | ba403dd6faf725d75c84d8be0891376c1364898719ce30b22d647bf581b1ac0d |
| `drivers/gpu/drm/amd/amdgpu/clearstate_defs.h` | 70eb9f9a57d2cacb2ee2d7b9f17485384019ac03c845ce4bea941acdae137285 |
| `drivers/gpu/drm/amd/amdgpu/clearstate_si.h` | 013698552542f8f7203c36ca644e5c89da4911f33fa1d5996c53d94e9a4db4b9 |
| `drivers/gpu/drm/amd/amdgpu/gfx_v6_0.c` | bfb3a74f91154a1fadb351cb38d3667a6912a4733b054aa33e56e19483cad281 |
| `drivers/gpu/drm/amd/amdgpu/gfx_v6_0.h` | 7b22c133eb8ac10721e904c09d54c2a379798d7b9e2fa555cbb6b7de640cdef2 |
| `drivers/gpu/drm/amd/amdgpu/gmc_v6_0.c` | 0ee5b23b9d796cfc51e1a5bca6719f0579c4dc4c08ad6cba4d374d874bc76a7b |
| `drivers/gpu/drm/amd/amdgpu/gmc_v6_0.h` | 10a90a8b0dd86a4c3de74db5ae07d492c258d2bb0e0de7c61910d1cc82b26882 |
| `drivers/gpu/drm/amd/amdgpu/si_dma.c` | 546c7e512f76d50557e8b81641c208378cdf4dd8942b05ad5f290f1e403dac65 |
| `drivers/gpu/drm/amd/amdgpu/si_dma.h` | a8284c2724250dd3267fd5c0f67a1ab3d7cd693315dd24a9261fc0fa2eb467f9 |
| `drivers/gpu/drm/amd/amdgpu/si_enums.h` | 5cba5181f355382430bc1cdf8b103f073efc644144a4890875b6c1b5488aa80f |
| `drivers/gpu/drm/amd/amdgpu/si_ih.c` | 274db49ab02c3101c16bbf994cea386c944d712b2e018caccec1273a65b5eb68 |
| `drivers/gpu/drm/amd/amdgpu/si_ih.h` | 9aadff9731fd5621024daa7f5dafda889747b45326ed154aef4b620f6b14de56 |
| `drivers/gpu/drm/amd/amdgpu/si.c` | 08c3adf1f27360149ee151cbf494b00d220b56e6ab33aee5a14f5d5c6cd76f26 |
| `drivers/gpu/drm/amd/amdgpu/si.h` | 2a01716f59255598a43e53aa91d9ee97a5bbe3cc7e6355397c49da1a100a31ec |
| `drivers/gpu/drm/amd/amdgpu/sid.h` | dd03da7ad3a67f92a6a4e28b8236f6706b39c33c0f1fdaec5951b17e1b8f25a5 |
| `drivers/gpu/drm/amd/include/amd_pcie.h` | 29fe4042b597af13e02eb6f86f823b6a2f4fc113aee11a699ca260a5873aa0f6 |
| `drivers/gpu/drm/amd/include/amd_shared.h` | 8d139d5ca40b769594fb26c8df3decff11f17384113382459366f17e1104511f |
| `drivers/gpu/drm/amd/include/asic_reg/bif/bif_3_0_d.h` | 0e6bee413f88c97be438f4f18c8f07a5dbb8e8039979c4c1607f461e866202d4 |
| `drivers/gpu/drm/amd/include/asic_reg/bif/bif_3_0_sh_mask.h` | cab80a88defb3359627eb78f607d3bfd978d1179c30333227702df433c22833d |
| `drivers/gpu/drm/amd/include/asic_reg/dce/dce_6_0_d.h` | 8e0e829fa89c434d05a998376251f90a2005b4fec298c6cd5efbdf1e09f5f208 |
| `drivers/gpu/drm/amd/include/asic_reg/dce/dce_6_0_sh_mask.h` | eb0a59f8b0fbc9cc44e1f3959265d594213ce02fabca0c2e0b6d4d6f614c1b49 |
| `drivers/gpu/drm/amd/include/asic_reg/gca/gfx_6_0_d.h` | 8a237c3dc570df593679b5bcb191a5670f7386bc125d8fbb9b5d149cb26320ad |
| `drivers/gpu/drm/amd/include/asic_reg/gca/gfx_6_0_sh_mask.h` | 8dc363c36bc9739a526019bbd1b46c6a8e1ae117fca346e6bcb1bfdf9861af29 |
| `drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_enum.h` | 4882624c0e6a13d726710a0c957981928522dacee8478200408a0cee5fba18cf |
| `drivers/gpu/drm/amd/include/asic_reg/gmc/gmc_6_0_d.h` | df71d476fdecb8a90ac6ce663469fe5bb72b6b3c179a68ef35fa5a7e9e031ad9 |
| `drivers/gpu/drm/amd/include/asic_reg/gmc/gmc_6_0_sh_mask.h` | f886b842a92781dac8722e7976a8d61273f656dde9aaa657eb3a9f7b46b965cb |
| `drivers/gpu/drm/amd/include/asic_reg/oss/oss_1_0_d.h` | fbd7e984d0da89a2f58a826a7cc2ecde62dc4d8b98998214eaf2b1af2c67911e |
| `drivers/gpu/drm/amd/include/asic_reg/oss/oss_1_0_sh_mask.h` | 76ade00fd4a84701e1afbd96cfec963ae3d60a53df7c94d845a007484a96ae13 |
| `drivers/gpu/drm/amd/include/asic_reg/smu/smu_6_0_d.h` | 37145e723f3a1a47a9ef1747886e1dfc57ab489223ad372ebfee6c7d5a08c22c |
| `drivers/gpu/drm/amd/include/asic_reg/smu/smu_6_0_sh_mask.h` | e1cd222baf465efe1e620e09b741e5bfc7a397a51c48ff338d0cba2e2de17f78 |
| `drivers/gpu/drm/amd/include/asic_reg/uvd/uvd_4_0_d.h` | 066d676702485335d5926e80c130984c99a75c025ed16934108624783e86000f |
| `drivers/gpu/drm/amd/include/atombios.h` | 2f543d5518d0a2bfd3a20c0220faac6edf1f1023279c952d503edffc02667dc0 |
| `drivers/gpu/drm/amd/include/soc_v1_0_ih_clientid.h` | 38e89c9097bc580feb143ec44e293b0ccb30bfbdc81bcea9841a7295698bae15 |
| `drivers/gpu/drm/amd/include/soc15_ih_clientid.h` | 9797586db00d3d6e0b080032a40b94e266f4290816164d2638602fbf12541ecd |
| `include/drm/amd_asic_type.h` | cea577df48a684246ef00d887350d42b9fa5048b9135ae5e8e93dc9eb94a7809 |
| `include/uapi/drm/amdgpu_drm.h` | ac7f6dae44b224f34591152123fbf2047d3486c35d39f3eb818a96ea47a6462a |
| `include/uapi/drm/drm_mode.h` | cfa0f3dff969178ec1f30abcb2072feb1a0b9f3aabb55580f3b08a77fd13e1e8 |
| `include/uapi/drm/drm.h` | 51fb1105a52c2085e6bdf4d6a3be347dfe8097d2375d8720d63106cc59f3fb6b |
