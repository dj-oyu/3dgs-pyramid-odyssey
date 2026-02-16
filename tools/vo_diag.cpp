// VO diagnostic: handle leftover state from previous runs
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <unistd.h>

extern "C" {
#include "ax_sys_api.h"
#include "ax_vo_api.h"
}

int main() {
    printf("=== VO Diagnostic v3 ===\n\n");

    AX_S32 ret = AX_SYS_Init();
    if (ret) { fprintf(stderr, "SYS_Init: 0x%08x\n", ret); return 1; }

    ret = AX_VO_Init();
    if (ret) { fprintf(stderr, "VO_Init: 0x%08x\n", ret); AX_SYS_Deinit(); return 1; }
    printf("[OK] Init\n\n");

    // Try to clean up any leftover state from previous runs
    printf("--- Cleanup stale state ---\n");
    for (int dev = 0; dev < 3; dev++) {
        ret = AX_VO_Disable(dev);
        printf("  Disable(%d): 0x%08x\n", dev, ret);
    }
    printf("\n");

    // Full Deinit/Init cycle to ensure clean state
    AX_VO_Deinit();
    ret = AX_VO_Init();
    if (ret) { fprintf(stderr, "VO_Init(2): 0x%08x\n", ret); AX_SYS_Deinit(); return 1; }
    printf("[OK] Re-Init after cleanup\n\n");

    // Now try each device
    for (int dev = 0; dev < 3; dev++) {
        printf("--- Device %d ---\n", dev);

        AX_VO_PUB_ATTR_T pub;
        memset(&pub, 0, sizeof(pub));
        pub.enMode = AX_VO_MODE_OFFLINE;
        pub.enIntfType = AX_VO_INTF_HDMI;
        pub.enIntfFmt = AX_VO_OUT_FMT_UNUSED;
        pub.enIntfSync = AX_VO_OUTPUT_1080P60;
        pub.stHdmiAttr.bEnableHdmi = AX_TRUE;
        pub.stHdmiAttr.enDefaultMode = AX_HDMI_FORCE_HDMI;
        pub.stHdmiAttr.eOutCscQuantization = AX_HDMI_QUANTIZATION_FULL_RANGE;

        ret = AX_VO_SetPubAttr(dev, &pub);
        printf("  SetPubAttr: 0x%08x\n", ret);
        if (ret) continue;

        ret = AX_VO_Enable(dev);
        printf("  Enable: 0x%08x\n", ret);
        if (ret) continue;

        printf("  [OK] Dev %d enabled!\n", dev);

        // Try layer: Create -> Bind -> SetAttr -> Enable
        VO_LAYER layer = 0;
        ret = AX_VO_CreateVideoLayer(&layer);
        printf("  CreateVideoLayer: 0x%08x (layer=%u)\n", ret, layer);

        if (ret == 0) {
            ret = AX_VO_BindVideoLayer(layer, dev);
            printf("  BindVideoLayer: 0x%08x\n", ret);

            if (ret == 0) {
                // Try multiple pixel formats
                AX_IMG_FORMAT_E fmts[] = {
                    AX_FORMAT_ARGB8888, AX_FORMAT_RGBA8888,
                    AX_FORMAT_BGRA8888, AX_FORMAT_RGB888,
                    AX_FORMAT_YUV420_SEMIPLANAR,
                };
                const char *names[] = {"ARGB8888","RGBA8888","BGRA8888","RGB888","NV12"};

                for (int fi = 0; fi < 5; fi++) {
                    AX_VO_VIDEO_LAYER_ATTR_T attr;
                    memset(&attr, 0, sizeof(attr));
                    attr.stDispRect = {0, 0, 1920, 1080};
                    attr.stImageSize = {1920, 1080};
                    attr.enPixFmt = fmts[fi];
                    attr.enSyncMode = AX_VO_LAYER_SYNC_NORMAL;
                    attr.u32FifoDepth = 2;
                    attr.u32BkClr = 0xFF000000;
                    attr.stCompressInfo.enCompressMode = AX_COMPRESS_MODE_NONE;
                    attr.enPartMode = AX_VO_PART_MODE_SINGLE;
                    attr.enBlendMode = AX_VO_BLEND_MODE_DEFAULT;
                    attr.enEngineMode = AX_VO_ENGINE_MODE_AUTO;
                    attr.u32PoolId = AX_INVALID_POOLID;
                    attr.f32FrmRate = 60.0f;

                    ret = AX_VO_SetVideoLayerAttr(layer, &attr);
                    printf("    SetLayerAttr(%s): 0x%08x\n", names[fi], ret);
                    if (ret == 0) {
                        ret = AX_VO_EnableVideoLayer(layer);
                        printf("    EnableLayer: 0x%08x\n", ret);
                        if (ret == 0) {
                            // Full test: channel + send frame
                            AX_VO_CHN_ATTR_T ca;
                            memset(&ca, 0, sizeof(ca));
                            ca.stRect = {0, 0, 1920, 1080};
                            ca.u32FifoDepth = 2;
                            ret = AX_VO_SetChnAttr(layer, 0, &ca);
                            printf("    SetChn: 0x%08x\n", ret);
                            ret = AX_VO_EnableChn(layer, 0);
                            printf("    EnableChn: 0x%08x\n", ret);

                            if (ret == 0) {
                                int bpp = (fmts[fi] == AX_FORMAT_RGB888) ? 3 : 4;
                                if (fmts[fi] == AX_FORMAT_YUV420_SEMIPLANAR) bpp = 0; // special
                                AX_U32 fb_size = 1920 * 1080 * (bpp ? bpp : 2);
                                AX_U64 phys = 0;
                                AX_VOID *virt = nullptr;
                                ret = AX_SYS_MemAlloc(&phys, &virt, fb_size, 4096, (const AX_S8*)"test");
                                if (ret == 0) {
                                    // Fill with visible pattern
                                    uint8_t *p = (uint8_t*)virt;
                                    for (uint32_t y = 0; y < 1080; y++)
                                        for (uint32_t x = 0; x < 1920; x++) {
                                            if (bpp == 4) {
                                                uint32_t *px = (uint32_t*)(p + (y*1920+x)*4);
                                                uint8_t r = (x * 255) / 1920;
                                                uint8_t g = (y * 255) / 1080;
                                                *px = 0xFF000000 | (r<<16) | (g<<8) | 0x80;
                                            } else {
                                                memset(p + (y*1920+x)*bpp, 0x80, bpp ? bpp : 1);
                                            }
                                        }

                                    AX_VIDEO_FRAME_T frame;
                                    memset(&frame, 0, sizeof(frame));
                                    frame.u32Width = 1920;
                                    frame.u32Height = 1080;
                                    frame.enImgFormat = fmts[fi];
                                    frame.u32PicStride[0] = 1920 * (bpp ? bpp : 1);
                                    frame.u64PhyAddr[0] = phys;
                                    frame.u64VirAddr[0] = (AX_U64)virt;
                                    frame.u32FrameSize = fb_size;

                                    ret = AX_VO_SendFrame(layer, 0, &frame, 500);
                                    printf("    SendFrame: 0x%08x\n", ret);
                                    if (ret == 0) {
                                        printf("    >>> SUCCESS - check HDMI (3s) <<<\n");
                                        sleep(3);
                                    }
                                    AX_SYS_MemFree(phys, virt);
                                }
                                AX_VO_DisableChn(layer, 0);
                            }
                            AX_VO_DisableVideoLayer(layer);
                        }
                        break;  // Found working format, stop trying
                    }
                }
                AX_VO_UnBindVideoLayer(layer, dev);
            }
            AX_VO_DestroyVideoLayer(layer);
        }

        AX_VO_Disable(dev);
    }

    AX_VO_Deinit();
    AX_SYS_Deinit();
    printf("\nDone.\n");
    return 0;
}
