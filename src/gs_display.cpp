#include "gs_display.h"
#include "gs_memory.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <arm_neon.h>

extern "C" {
#include "ax_vo_api.h"
}

// ============================================================
// Framebuffer backend (/dev/fb)
// ============================================================

static bool fb_init(DisplayContext &ctx) {
    // Try fb0 through fb7
    for (int i = 0; i < 8; i++) {
        char path[32];
        snprintf(path, sizeof(path), "/dev/fb%d", i);

        int fd = open(path, O_RDWR);
        if (fd < 0) continue;

        struct fb_var_screeninfo vinfo;
        struct fb_fix_screeninfo finfo;

        if (ioctl(fd, FBIOGET_VSCREENINFO, &vinfo) < 0 ||
            ioctl(fd, FBIOGET_FSCREENINFO, &finfo) < 0) {
            close(fd);
            continue;
        }

        printf("[gs_display] %s: %ux%u (virtual %ux%u), %u bpp, stride=%u, memlen=%u\n",
               path, vinfo.xres, vinfo.yres,
               vinfo.xres_virtual, vinfo.yres_virtual,
               vinfo.bits_per_pixel, finfo.line_length, finfo.smem_len);
        printf("[gs_display]   offset: x=%u y=%u\n", vinfo.xoffset, vinfo.yoffset);
        printf("[gs_display]   RGBA offsets: R(%u,%u) G(%u,%u) B(%u,%u) A(%u,%u)\n",
               vinfo.red.offset, vinfo.red.length,
               vinfo.green.offset, vinfo.green.length,
               vinfo.blue.offset, vinfo.blue.length,
               vinfo.transp.offset, vinfo.transp.length);

        // Skip tiny or unusable framebuffers
        if (vinfo.xres < 640 || vinfo.yres < 480 || finfo.smem_len == 0) {
            close(fd);
            continue;
        }

        // Use this framebuffer
        uint32_t mem_size = finfo.smem_len;
        void *mem = mmap(nullptr, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mem == MAP_FAILED) {
            fprintf(stderr, "[gs_display] mmap %s failed\n", path);
            close(fd);
            continue;
        }

        // Ensure yoffset=0 so we display from the start of the buffer
        if (vinfo.yoffset != 0 || vinfo.xoffset != 0) {
            vinfo.xoffset = 0;
            vinfo.yoffset = 0;
            ioctl(fd, FBIOPUT_VSCREENINFO, &vinfo);
        }

        ctx.fb_fd = fd;
        ctx.fb_mem = static_cast<uint8_t*>(mem);
        ctx.fb_mem_size = mem_size;
        ctx.fb_stride = finfo.line_length;
        ctx.fb_bpp = vinfo.bits_per_pixel;
        ctx.width = vinfo.xres;
        ctx.height = vinfo.yres;
        ctx.backend = DisplayBackend::Framebuffer;
        ctx.initialized = true;

        // Determine pixel byte order from vscreeninfo
        // Standard ARGB8888: A@24, R@16, G@8, B@0
        // If B is at offset 16 it's ABGR (need to swap R/B)
        ctx.fb_swap_rb = (vinfo.blue.offset == 16);

        // Clear entire fb memory to opaque black (alpha=0xFF)
        // This prevents transparency artifacts from the overlay layer
        memset(ctx.fb_mem, 0, mem_size);
        // Set alpha to 0xFF for all pixels in visible area
        uint32_t *pixels = reinterpret_cast<uint32_t*>(ctx.fb_mem);
        uint32_t total_pixels = mem_size / 4;
        for (uint32_t p = 0; p < total_pixels; p++) {
            pixels[p] = 0xFF000000;  // Opaque black
        }

        printf("[gs_display] Using %s: %ux%u %ubpp swap_rb=%d (framebuffer backend)\n",
               path, ctx.width, ctx.height, ctx.fb_bpp, ctx.fb_swap_rb);
        return true;
    }

    fprintf(stderr, "[gs_display] No usable framebuffer found\n");
    return false;
}

static void fb_deinit(DisplayContext &ctx) {
    if (ctx.fb_mem && ctx.fb_mem != MAP_FAILED) {
        memset(ctx.fb_mem, 0, ctx.fb_mem_size);
        munmap(ctx.fb_mem, ctx.fb_mem_size);
    }
    if (ctx.fb_fd >= 0) close(ctx.fb_fd);
    ctx.fb_fd = -1;
    ctx.fb_mem = nullptr;
}

static bool fb_send_frame(DisplayContext &ctx, const Framebuffer &fb) {
    if (!ctx.fb_mem) return false;

    if (ctx.fb_bpp == 32 && fb.width == ctx.width && fb.height == ctx.height) {
        // Same resolution path
        if (!ctx.fb_swap_rb && fb.stride == ctx.fb_stride) {
            // Fast path: direct memcpy
            uint32_t copy_bytes = fb.height * ctx.fb_stride;
            if (copy_bytes <= ctx.fb_mem_size) {
                memcpy(ctx.fb_mem, fb.data, copy_bytes);
            }
        } else {
            // Row-by-row copy
            uint32_t row_bytes = fb.width * 4;
            for (uint32_t y = 0; y < fb.height; y++) {
                memcpy(ctx.fb_mem + y * ctx.fb_stride,
                       fb.data + y * fb.stride, row_bytes);
            }
        }
    } else if (ctx.fb_bpp == 32 && fb.width <= ctx.width && fb.height <= ctx.height) {
        // Upscale path: nearest-neighbor directly into fb memory
        // Uses fixed-point arithmetic for fast scaling
        uint32_t sx_step = (fb.width  << 16) / ctx.width;
        uint32_t sy_step = (fb.height << 16) / ctx.height;

        // Write row-by-row to fb_mem (sequential writes are important for uncached memory)
        for (uint32_t dy = 0; dy < ctx.height; dy++) {
            uint32_t sy = (dy * sy_step) >> 16;
            const uint32_t *src_row = reinterpret_cast<const uint32_t*>(
                fb.data + sy * fb.stride);
            uint32_t *dst_row = reinterpret_cast<uint32_t*>(
                ctx.fb_mem + dy * ctx.fb_stride);

            // For integer scale factors, use NEON-accelerated pixel duplication
            if (fb.width * 2 == ctx.width) {
                // 2x horizontal: NEON zip to duplicate 4 pixels → 8 pixels
                uint32_t sx = 0;
                for (; sx + 4 <= fb.width; sx += 4) {
                    uint32x4_t p = vld1q_u32(&src_row[sx]);
                    uint32x4x2_t dup = vzipq_u32(p, p);
                    vst1q_u32(&dst_row[sx * 2],     dup.val[0]);
                    vst1q_u32(&dst_row[sx * 2 + 4], dup.val[1]);
                }
                for (; sx < fb.width; sx++) {
                    uint32_t pixel = src_row[sx];
                    dst_row[sx * 2]     = pixel;
                    dst_row[sx * 2 + 1] = pixel;
                }
            } else if (fb.width * 4 == ctx.width) {
                // 4x horizontal: each NEON load → 4 stores
                uint32_t sx = 0;
                for (; sx + 4 <= fb.width; sx += 4) {
                    uint32x4_t p = vld1q_u32(&src_row[sx]);
                    // Duplicate each lane to a full register
                    uint32x4_t p0 = vdupq_laneq_u32(p, 0);
                    uint32x4_t p1 = vdupq_laneq_u32(p, 1);
                    uint32x4_t p2 = vdupq_laneq_u32(p, 2);
                    uint32x4_t p3 = vdupq_laneq_u32(p, 3);
                    vst1q_u32(&dst_row[sx * 4],      p0);
                    vst1q_u32(&dst_row[sx * 4 + 4],  p1);
                    vst1q_u32(&dst_row[sx * 4 + 8],  p2);
                    vst1q_u32(&dst_row[sx * 4 + 12], p3);
                }
                for (; sx < fb.width; sx++) {
                    uint32_t pixel = src_row[sx];
                    uint32_t dx = sx * 4;
                    dst_row[dx]     = pixel;
                    dst_row[dx + 1] = pixel;
                    dst_row[dx + 2] = pixel;
                    dst_row[dx + 3] = pixel;
                }
            } else {
                // General case: fixed-point scaling
                uint32_t sx_acc = 0;
                for (uint32_t dx = 0; dx < ctx.width; dx++) {
                    dst_row[dx] = src_row[sx_acc >> 16];
                    sx_acc += sx_step;
                }
            }
        }
    } else if (ctx.fb_bpp == 32) {
        // Downscale or partial copy
        uint32_t copy_h = (fb.height < ctx.height) ? fb.height : ctx.height;
        uint32_t copy_w = (fb.width < ctx.width) ? fb.width : ctx.width;
        uint32_t row_bytes = copy_w * 4;
        for (uint32_t y = 0; y < copy_h; y++) {
            memcpy(ctx.fb_mem + y * ctx.fb_stride,
                   fb.data + y * fb.stride, row_bytes);
        }
    }

    // Try vsync if enabled (ignore error if not supported)
    if (ctx.fb_vsync) {
        int zero = 0;
        ioctl(ctx.fb_fd, FBIO_WAITFORVSYNC, &zero);
    }

    return true;
}

// ============================================================
// VO backend (AX_VO API)
// ============================================================

static bool vo_init(DisplayContext &ctx) {
    AX_S32 ret;

    ret = AX_VO_Init();
    if (ret != 0) {
        fprintf(stderr, "[gs_display] AX_VO_Init failed: 0x%08x\n", ret);
        return false;
    }

    AX_VO_PUB_ATTR_T pub_attr;
    memset(&pub_attr, 0, sizeof(pub_attr));
    pub_attr.enMode     = AX_VO_MODE_OFFLINE;
    pub_attr.enIntfType = AX_VO_INTF_HDMI;
    pub_attr.enIntfFmt  = AX_VO_OUT_FMT_UNUSED;
    pub_attr.enIntfSync = AX_VO_OUTPUT_1080P60;
    pub_attr.stHdmiAttr.bEnableHdmi = AX_TRUE;
    pub_attr.stHdmiAttr.enDefaultMode = AX_HDMI_FORCE_HDMI;
    pub_attr.stHdmiAttr.eOutCscQuantization = AX_HDMI_QUANTIZATION_FULL_RANGE;

    ret = AX_VO_SetPubAttr(ctx.vo_dev, &pub_attr);
    if (ret != 0) {
        fprintf(stderr, "[gs_display] AX_VO_SetPubAttr(dev=%u) failed: 0x%08x\n", ctx.vo_dev, ret);
        AX_VO_Deinit();
        return false;
    }

    ret = AX_VO_Enable(ctx.vo_dev);
    if (ret != 0) {
        fprintf(stderr, "[gs_display] AX_VO_Enable failed: 0x%08x\n", ret);
        AX_VO_Deinit();
        return false;
    }

    VO_LAYER layer = 0;
    ret = AX_VO_CreateVideoLayer(&layer);
    if (ret != 0) {
        fprintf(stderr, "[gs_display] AX_VO_CreateVideoLayer failed: 0x%08x\n", ret);
        AX_VO_Disable(ctx.vo_dev);
        AX_VO_Deinit();
        return false;
    }
    ctx.vo_layer = layer;

    ret = AX_VO_BindVideoLayer(ctx.vo_layer, ctx.vo_dev);
    if (ret != 0) {
        fprintf(stderr, "[gs_display] AX_VO_BindVideoLayer failed: 0x%08x\n", ret);
        AX_VO_DestroyVideoLayer(ctx.vo_layer);
        AX_VO_Disable(ctx.vo_dev);
        AX_VO_Deinit();
        return false;
    }

    AX_VO_VIDEO_LAYER_ATTR_T layer_attr;
    memset(&layer_attr, 0, sizeof(layer_attr));
    layer_attr.stDispRect.u32Width = ctx.width;
    layer_attr.stDispRect.u32Height = ctx.height;
    layer_attr.stImageSize.u32Width = ctx.width;
    layer_attr.stImageSize.u32Height = ctx.height;
    layer_attr.enPixFmt = AX_FORMAT_ARGB8888;
    layer_attr.enSyncMode = AX_VO_LAYER_SYNC_NORMAL;
    layer_attr.u32FifoDepth = 2;
    layer_attr.u32BkClr = 0xFF000000;
    layer_attr.stCompressInfo.enCompressMode = AX_COMPRESS_MODE_NONE;
    layer_attr.enPartMode = AX_VO_PART_MODE_SINGLE;
    layer_attr.enBlendMode = AX_VO_BLEND_MODE_DEFAULT;
    layer_attr.enEngineMode = AX_VO_ENGINE_MODE_AUTO;
    layer_attr.u32PoolId = AX_INVALID_POOLID;
    layer_attr.f32FrmRate = 60.0f;

    ret = AX_VO_SetVideoLayerAttr(ctx.vo_layer, &layer_attr);
    if (ret != 0) {
        fprintf(stderr, "[gs_display] AX_VO_SetVideoLayerAttr failed: 0x%08x\n", ret);
        AX_VO_UnBindVideoLayer(ctx.vo_layer, ctx.vo_dev);
        AX_VO_DestroyVideoLayer(ctx.vo_layer);
        AX_VO_Disable(ctx.vo_dev);
        AX_VO_Deinit();
        return false;
    }

    ret = AX_VO_EnableVideoLayer(ctx.vo_layer);
    if (ret != 0) {
        fprintf(stderr, "[gs_display] AX_VO_EnableVideoLayer failed: 0x%08x\n", ret);
        AX_VO_UnBindVideoLayer(ctx.vo_layer, ctx.vo_dev);
        AX_VO_DestroyVideoLayer(ctx.vo_layer);
        AX_VO_Disable(ctx.vo_dev);
        AX_VO_Deinit();
        return false;
    }

    ctx.vo_chn = 0;
    AX_VO_CHN_ATTR_T chn_attr;
    memset(&chn_attr, 0, sizeof(chn_attr));
    chn_attr.stRect.u32Width = ctx.width;
    chn_attr.stRect.u32Height = ctx.height;
    chn_attr.u32FifoDepth = 2;
    chn_attr.bKeepPrevFr = AX_TRUE;

    AX_VO_SetChnAttr(ctx.vo_layer, ctx.vo_chn, &chn_attr);
    AX_VO_EnableChn(ctx.vo_layer, ctx.vo_chn);

    ctx.backend = DisplayBackend::VO;
    ctx.initialized = true;
    printf("[gs_display] VO backend: dev=%u layer=%u %ux%u\n",
           ctx.vo_dev, ctx.vo_layer, ctx.width, ctx.height);
    return true;
}

static void vo_deinit(DisplayContext &ctx) {
    AX_VO_DisableChn(ctx.vo_layer, ctx.vo_chn);
    AX_VO_DisableVideoLayer(ctx.vo_layer);
    AX_VO_UnBindVideoLayer(ctx.vo_layer, ctx.vo_dev);
    AX_VO_DestroyVideoLayer(ctx.vo_layer);
    AX_VO_Disable(ctx.vo_dev);
    AX_VO_Deinit();
}

static bool vo_send_frame(DisplayContext &ctx, const Framebuffer &fb) {
    AX_VIDEO_FRAME_T frame;
    memset(&frame, 0, sizeof(frame));
    frame.u32Width  = fb.width;
    frame.u32Height = fb.height;
    frame.enImgFormat = AX_FORMAT_ARGB8888;
    frame.u32PicStride[0] = fb.stride;
    frame.u64PhyAddr[0] = fb.phys_addr;
    frame.u64VirAddr[0] = reinterpret_cast<AX_U64>(fb.data);
    frame.u32FrameSize = fb.size;

    AX_S32 ret = AX_VO_SendFrame(ctx.vo_layer, ctx.vo_chn, &frame, -1);
    return ret == 0;
}

// ============================================================
// Public API
// ============================================================

bool gs_display_init(DisplayContext &ctx) {
    // Try VO first
    printf("[gs_display] Trying VO backend (dev=%u)...\n", ctx.vo_dev);
    if (vo_init(ctx)) {
        return true;
    }

    // Fallback to framebuffer
    printf("[gs_display] VO failed, trying framebuffer backend...\n");
    return fb_init(ctx);
}

void gs_display_deinit(DisplayContext &ctx) {
    if (!ctx.initialized) return;

    if (ctx.backend == DisplayBackend::VO) {
        vo_deinit(ctx);
    } else {
        fb_deinit(ctx);
    }

    ctx.initialized = false;
    printf("[gs_display] Display deinitialized\n");
}

bool gs_display_send_frame(DisplayContext &ctx, const Framebuffer &fb) {
    if (!ctx.initialized) return false;

    if (ctx.backend == DisplayBackend::VO) {
        return vo_send_frame(ctx, fb);
    } else {
        return fb_send_frame(ctx, fb);
    }
}

bool gs_framebuffer_alloc(Framebuffer &fb, uint32_t width, uint32_t height, bool need_phys) {
    fb.data      = nullptr;
    fb.phys_addr = 0;
    fb.width     = width;
    fb.height    = height;
    fb.stride    = width * 4;  // ARGB8888
    fb.size      = fb.stride * height;

    if (need_phys) {
        // CMM allocation (uncached, but provides physical address for VO)
        CmmBuffer buf;
        if (gs_cmm_alloc(buf, fb.size, "framebuffer")) {
            fb.data      = reinterpret_cast<uint8_t*>(buf.virt_addr);
            fb.phys_addr = buf.phys_addr;
        }
    }

    if (!fb.data) {
        // Cached regular memory (much faster for CPU pixel writes)
        void *ptr = nullptr;
        if (posix_memalign(&ptr, 4096, fb.size) != 0 || !ptr) {
            fprintf(stderr, "[gs_display] Failed to allocate framebuffer %ux%u (%u bytes)\n",
                    width, height, fb.size);
            return false;
        }
        fb.data = static_cast<uint8_t*>(ptr);
        fb.phys_addr = 0;
    }

    memset(fb.data, 0, fb.size);
    printf("[gs_display] Framebuffer: %ux%u stride=%u size=%u phys=0x%llx\n",
           width, height, fb.stride, fb.size, (unsigned long long)fb.phys_addr);
    return true;
}

void gs_framebuffer_free(Framebuffer &fb) {
    if (fb.data) {
        if (fb.phys_addr) {
            CmmBuffer buf;
            buf.virt_addr = fb.data;
            buf.phys_addr = fb.phys_addr;
            buf.size = fb.size;
            gs_cmm_free(buf);
        } else {
            free(fb.data);
        }
    }
    fb.data = nullptr;
    fb.phys_addr = 0;
    fb.size = 0;
}
