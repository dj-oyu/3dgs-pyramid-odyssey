/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Shanghai) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Shanghai) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Shanghai) Co., Ltd.
 *
 **************************************************************************************************/

#ifndef _AX_GLOBAL_TYPE_H_
#define _AX_GLOBAL_TYPE_H_
#include "ax_base_type.h"

#define DEF_ALL_MOD_GRP_MAX         (164)
#define DEF_ALL_MOD_CHN_MAX         (128)
#define AX_LINK_DEST_MAXNUM         (6)
#define AX_MAX_COLOR_COMPONENT      (3)
#define AX_MAX_COMPRESS_LOSSY_LEVEL (10)
#define AX_INVALID_ID               (-1U)

#ifndef AX_SUCCESS
#define AX_SUCCESS                  (0)
#endif

#define AX_INVALID_FRMRATE          (0.0f)
typedef struct axFRAME_RATE_CTRL_T {
    AX_F32  fSrcFrameRate;
    AX_F32  fDstFrameRate;
} AX_FRAME_RATE_CTRL_T;

typedef enum {
    SYS_LOG_MIN         = -1,
    SYS_LOG_EMERGENCY   = 0,
    SYS_LOG_ALERT       = 1,
    SYS_LOG_CRITICAL    = 2,
    SYS_LOG_ERROR       = 3,
    SYS_LOG_WARN        = 4,
    SYS_LOG_NOTICE      = 5,
    SYS_LOG_INFO        = 6,
    SYS_LOG_DEBUG       = 7,
    SYS_LOG_MAX
} AX_LOG_LEVEL_E;

typedef enum {
    SYS_LOG_TARGET_MIN = 0,
    SYS_LOG_TARGET_STDERR = 1,
    SYS_LOG_TARGET_SYSLOG = 2,
    SYS_LOG_TARGET_NULL   = 3,
    SYS_LOG_TARGET_MAX
} AX_LOG_TARGET_E;

typedef enum
{
    PT_PCMU             = 0,
    PT_1016             = 1,
    PT_G721             = 2,
    PT_GSM              = 3,
    PT_G723             = 4,
    PT_DVI4_8K          = 5,
    PT_DVI4_16K         = 6,
    PT_LPC              = 7,
    PT_PCMA             = 8,
    PT_G722             = 9,
    PT_S16BE_STEREO     = 10,
    PT_S16BE_MONO       = 11,
    PT_QCELP            = 12,
    PT_CN               = 13,
    PT_MPEGAUDIO        = 14,
    PT_G728             = 15,
    PT_DVI4_3           = 16,
    PT_DVI4_4           = 17,
    PT_G729             = 18,
    PT_G711A            = 19,
    PT_G711U            = 20,
    PT_G726             = 21,
    PT_G729A            = 22,
    PT_LPCM             = 23,
    PT_CelB             = 25,
    PT_JPEG             = 26,
    PT_CUSM             = 27,
    PT_NV               = 28,
    PT_PICW             = 29,
    PT_CPV              = 30,
    PT_H261             = 31,
    PT_MPEGVIDEO        = 32,
    PT_MPEG2TS          = 33,
    PT_H263             = 34,
    PT_SPEG             = 35,
    PT_MPEG2VIDEO       = 36,
    PT_AAC              = 37,
    PT_WMA9STD          = 38,
    PT_HEAAC            = 39,
    PT_PCM_VOICE        = 40,
    PT_PCM_AUDIO        = 41,
    PT_AACLC            = 42,
    PT_MP3              = 43,
    PT_ADPCMA           = 49,
    PT_AEC              = 50,
    PT_X_LD             = 95,
    PT_H264             = 96,
    PT_D_GSM_HR         = 200,
    PT_D_GSM_EFR        = 201,
    PT_D_L8             = 202,
    PT_D_RED            = 203,
    PT_D_VDVI           = 204,
    PT_D_BT656          = 220,
    PT_D_H263_1998      = 221,
    PT_D_MP1S           = 222,
    PT_D_MP2P           = 223,
    PT_D_BMPEG          = 224,
    PT_MP4VIDEO         = 230,
    PT_MP4AUDIO         = 237,
    PT_VC1              = 238,
    PT_JVC_ASF          = 255,
    PT_D_AVI            = 256,
    PT_DIVX3            = 257,
    PT_AVS              = 258,
    PT_REAL8            = 259,
    PT_REAL9            = 260,
    PT_VP6              = 261,
    PT_VP6F             = 262,
    PT_VP6A             = 263,
    PT_SORENSON         = 264,
    PT_H265             = 265,
    PT_VP8              = 266,
    PT_MVC              = 267,
    PT_PNG              = 268,
    PT_AVS2             = 269,
    PT_VP7              = 270,
    PT_VP9              = 271,
    PT_AMR              = 1001,
    PT_MJPEG            = 1002,
    PT_AMRWB            = 1003,
    PT_PRORES           = 1006,
    PT_OPUS             = 1007,
    PT_BUTT
} AX_PAYLOAD_TYPE_E;

typedef enum {
    AX_VSCAN_FORMAT_RASTER = 0,
    AX_VSCAN_FORMAT_BUTT
} AX_VSCAN_FORMAT_E;

typedef enum
{
    AX_COMPRESS_MODE_NONE = 0,
    AX_COMPRESS_MODE_LOSSLESS,
    AX_COMPRESS_MODE_LOSSY,
    AX_COMPRESS_MODE_BUTT
} AX_COMPRESS_MODE_E;

typedef struct axFRAME_COMPRESS_INFO_T {
    AX_COMPRESS_MODE_E enCompressMode;
    AX_U32    u32CompressLevel;
} AX_FRAME_COMPRESS_INFO_T;

typedef enum axDYNAMIC_RANGE_E
{
    AX_DYNAMIC_RANGE_SDR8 = 0,
    AX_DYNAMIC_RANGE_SDR10,
    AX_DYNAMIC_RANGE_HDR10,
    AX_DYNAMIC_RANGE_HLG,
    AX_DYNAMIC_RANGE_SLF,
    AX_DYNAMIC_RANGE_XDR,
    AX_DYNAMIC_RANGE_BUTT
} AX_DYNAMIC_RANGE_E;

typedef enum axCOLOR_GAMUT_E
{
    AX_COLOR_GAMUT_BT601 = 0,
    AX_COLOR_GAMUT_BT709,
    AX_COLOR_GAMUT_BT2020,
    AX_COLOR_GAMUT_USER,
    AX_COLOR_GAMUT_BUTT
} AX_COLOR_GAMUT_E;

typedef enum
{
    AX_FORMAT_INVALID                               = -1,
    AX_FORMAT_YUV400                                = 0x0,
    AX_FORMAT_YUV420_PLANAR                         = 0x1,
    AX_FORMAT_YUV420_PLANAR_VU                      = 0x2,
    AX_FORMAT_YUV420_SEMIPLANAR                     = 0x3,
    AX_FORMAT_YUV420_SEMIPLANAR_VU                  = 0x4,
    AX_FORMAT_YUV422_PLANAR                         = 0x8,
    AX_FORMAT_YUV422_PLANAR_VU                      = 0x9,
    AX_FORMAT_YUV422_SEMIPLANAR                     = 0xA,
    AX_FORMAT_YUV422_SEMIPLANAR_VU                  = 0xB,
    AX_FORMAT_YUV422_INTERLEAVED_YUVY               = 0xC,
    AX_FORMAT_YUV422_INTERLEAVED_YUYV               = 0xD,
    AX_FORMAT_YUV422_INTERLEAVED_UYVY               = 0xE,
    AX_FORMAT_YUV422_INTERLEAVED_VYUY               = 0xF,
    AX_FORMAT_YUV422_INTERLEAVED_YVYU               = 0x10,
    AX_FORMAT_YUV444_PLANAR                         = 0x14,
    AX_FORMAT_YUV444_PLANAR_VU                      = 0x15,
    AX_FORMAT_YUV444_SEMIPLANAR                     = 0x16,
    AX_FORMAT_YUV444_SEMIPLANAR_VU                  = 0x17,
    AX_FORMAT_YUV444_PACKED                         = 0x18,
    AX_FORMAT_YUV400_10BIT                          = 0x20,
    AX_FORMAT_YUV420_PLANAR_10BIT_UV_PACKED_4Y5B    = 0x24,
    AX_FORMAT_YUV420_PLANAR_10BIT_I010              = 0x25,
    AX_FORMAT_YUV420_SEMIPLANAR_10BIT_P101010       = 0x28,
    AX_FORMAT_YUV420_SEMIPLANAR_10BIT_P010          = 0x2A,
    AX_FORMAT_YUV420_SEMIPLANAR_10BIT_P016          = 0x2C,
    AX_FORMAT_YUV420_SEMIPLANAR_10BIT_I016          = 0x2E,
    AX_FORMAT_YUV420_SEMIPLANAR_10BIT_12P16B        = 0x2F,
    AX_FORMAT_YUV444_PACKED_10BIT_P010              = 0x30,
    AX_FORMAT_YUV444_PACKED_10BIT_P101010           = 0x32,
    AX_FORMAT_YUV422_SEMIPLANAR_10BIT_P101010       = 0x33,
    AX_FORMAT_YUV422_SEMIPLANAR_10BIT_P010          = 0x34,
    AX_FORMAT_BAYER_RAW_8BPP                        = 0x80,
    AX_FORMAT_BAYER_RAW_10BPP                       = 0x81,
    AX_FORMAT_BAYER_RAW_12BPP                       = 0x82,
    AX_FORMAT_BAYER_RAW_14BPP                       = 0x83,
    AX_FORMAT_BAYER_RAW_16BPP                       = 0x84,
    AX_FORMAT_BAYER_RAW_10BPP_PACKED                = 0x85,
    AX_FORMAT_BAYER_RAW_12BPP_PACKED                = 0x86,
    AX_FORMAT_BAYER_RAW_14BPP_PACKED                = 0x87,
    AX_FORMAT_RGB565                                = 0xA0,
    AX_FORMAT_RGB888                                = 0xA1,
    AX_FORMAT_KRGB444                               = 0xA2,
    AX_FORMAT_KRGB555                               = 0xA3,
    AX_FORMAT_KRGB888                               = 0xA4,
    AX_FORMAT_BGR888                                = 0xA5,
    AX_FORMAT_BGR565                                = 0xA6,
    AX_FORMAT_ARGB4444                              = 0xC5,
    AX_FORMAT_ARGB1555                              = 0xC6,
    AX_FORMAT_ARGB8888                              = 0xC7,
    AX_FORMAT_ARGB8565                              = 0xC8,
    AX_FORMAT_RGBA8888                              = 0xC9,
    AX_FORMAT_RGBA5551                              = 0xCA,
    AX_FORMAT_RGBA4444                              = 0xCB,
    AX_FORMAT_RGBA5658                              = 0xCC,
    AX_FORMAT_ABGR4444                              = 0xCD,
    AX_FORMAT_ABGR1555                              = 0xCE,
    AX_FORMAT_ABGR8888                              = 0xCF,
    AX_FORMAT_ABGR8565                              = 0xD0,
    AX_FORMAT_BGRA8888                              = 0xD1,
    AX_FORMAT_BGRA5551                              = 0xD2,
    AX_FORMAT_BGRA4444                              = 0xD3,
    AX_FORMAT_BGRA5658                              = 0xD4,
    AX_FORMAT_BITMAP                                = 0xE0,
    AX_FORMAT_MAX
} AX_IMG_FORMAT_E;

typedef enum {
    AX_FRM_FLG_NONE  = 0x0,
    AX_FRM_FLG_USR_PIC  = (0x1 << 0),
    AX_FRM_FLG_FR_CTRL  = (0x1 << 1),
    AX_FRM_FLG_BUTT
} AX_FRAME_FLAG_E;

typedef struct axVIDEO_FRAME_T {
    AX_U32              u32Width;
    AX_U32              u32Height;
    AX_IMG_FORMAT_E     enImgFormat;
    AX_VSCAN_FORMAT_E   enVscanFormat;
    AX_FRAME_COMPRESS_INFO_T  stCompressInfo;
    AX_DYNAMIC_RANGE_E  stDynamicRange;
    AX_COLOR_GAMUT_E    stColorGamut;
    AX_U32              u32PicStride[AX_MAX_COLOR_COMPONENT];
    AX_U32              u32ExtStride[AX_MAX_COLOR_COMPONENT];
    AX_U64              u64PhyAddr[AX_MAX_COLOR_COMPONENT];
    AX_U64              u64VirAddr[AX_MAX_COLOR_COMPONENT];
    AX_U64              u64ExtPhyAddr[AX_MAX_COLOR_COMPONENT];
    AX_U64              u64ExtVirAddr[AX_MAX_COLOR_COMPONENT];
    AX_U32              u32HeaderSize[AX_MAX_COLOR_COMPONENT];
    AX_U32              u32BlkId[AX_MAX_COLOR_COMPONENT];
    AX_S16              s16CropX;
    AX_S16              s16CropY;
    AX_S16              s16CropWidth;
    AX_S16              s16CropHeight;
    AX_U32              u32TimeRef;
    AX_U64              u64PTS;
    AX_U64              u64SeqNum;
    AX_U64              u64UserData;
    AX_U64              u64PrivateData;
    AX_U32              u32FrameFlag;
    AX_U32              u32FrameSize;
} AX_VIDEO_FRAME_T;

typedef enum
{
    AX_ID_MIN      = 0x00,
    AX_ID_ISP      = 0x01,
    AX_ID_CE       = 0x02,
    AX_ID_VO       = 0x03,
    AX_ID_VDSP     = 0x04,
    AX_ID_EFUSE    = 0x05,
    AX_ID_NPU      = 0x06,
    AX_ID_VENC     = 0x07,
    AX_ID_VDEC     = 0x08,
    AX_ID_JENC     = 0x09,
    AX_ID_JDEC     = 0x0a,
    AX_ID_SYS      = 0x0b,
    AX_ID_AENC     = 0x0c,
    AX_ID_IVPS     = 0x0d,
    AX_ID_MIPI     = 0x0e,
    AX_ID_ADEC     = 0x0f,
    AX_ID_DMA      = 0x10,
    AX_ID_VIN      = 0x11,
    AX_ID_USER     = 0x12,
    AX_ID_IVES     = 0x13,
    AX_ID_SKEL     = 0x14,
    AX_ID_IVE      = 0x15,
    AX_ID_AVS      = 0x16,
    AX_ID_AVSCALI  = 0x17,
    AX_ID_3A       = 0X19,
    AX_ID_AUDIO    = 0x1a,
    AX_ID_PYRALITE = 0x1b,
    AX_ID_SIF      = 0x1c,
    AX_ID_AI       = 0X20,
    AX_ID_AO       = 0X21,
    AX_ID_SENSOR   = 0x22,
    AX_ID_NT       = 0x23,
    AX_ID_TDP      = 0X24,
    AX_ID_VPP      = 0X25,
    AX_ID_VGP      = 0X26,
    AX_ID_GDC      = 0x27,
    AX_ID_BASE     = 0x28,
    AX_ID_RESERVE  = 0x29,
    AX_ID_BUTT,
    AX_ID_CUST_MIN = 0x80,
    AX_ID_MAX      = 0xFF
} AX_MOD_ID_E;

typedef enum
{
    AX_UNLINK_MODE = 0,
    AX_LINK_MODE = 1,
} AX_LINK_MODE_E;

typedef enum axAUDIO_BIT_WIDTH_E {
    AX_AUDIO_BIT_WIDTH_8   = 0,
    AX_AUDIO_BIT_WIDTH_16  = 1,
    AX_AUDIO_BIT_WIDTH_24  = 2,
    AX_AUDIO_BIT_WIDTH_32  = 3,
    AX_AUDIO_BIT_WIDTH_BUTT,
} AX_AUDIO_BIT_WIDTH_E;

typedef enum axAUDIO_SOUND_MODE_E {
    AX_AUDIO_SOUND_MODE_MONO   = 0,
    AX_AUDIO_SOUND_MODE_STEREO = 1,
    AX_AUDIO_SOUND_MODE_BUTT
} AX_AUDIO_SOUND_MODE_E;

typedef struct axAUDIO_FRAME_T {
    AX_AUDIO_BIT_WIDTH_E   enBitwidth;
    AX_AUDIO_SOUND_MODE_E  enSoundmode;
    AX_U8  *u64VirAddr;
    AX_U64  u64PhyAddr;
    AX_U64  u64TimeStamp;
    AX_U32  u32Seq;
    AX_U32  u32Len;
    AX_U32  u32PoolId[2];
    AX_BOOL bEof;
    AX_U32 u32BlkId;
} AX_AUDIO_FRAME_T;

typedef struct axAUDIO_FRAME_INFO_T {
    AX_AUDIO_FRAME_T  stAFrame;
    AX_MOD_ID_E         enModId;
    AX_BOOL             bEndOfStream;
} AX_AUDIO_FRAME_INFO_T;

typedef struct axVIDEO_FRAME_INFO_T {
    AX_VIDEO_FRAME_T    stVFrame;
    AX_MOD_ID_E         enModId;
    AX_BOOL             bEndOfStream;
} AX_VIDEO_FRAME_INFO_T;

typedef enum {
    AX_NOTIFY_EVENT_SLEEP   = 0,
    AX_NOTIFY_EVENT_WAKEUP  = 1,
    AX_NOTIFY_EVENT_MAX
} AX_NOTIFY_EVENT_E;

typedef enum {
    AX_SYS_CLK_HIGH_MODE             = 0,
    AX_SYS_CLK_HIGH_HOTBALANCE_MODE  = 1,
    AX_SYS_CLK_MID_MODE              = 2,
    AX_SYS_CLK_MID_HOTBALANCE_MODE   = 3,
    AX_SYS_CLK_MAX_MODE              = 4,
} AX_SYS_CLK_LEVEL_E;

typedef enum {
    AX_CPU_CLK_ID       = 0,
    AX_BUS_CLK_ID       = 1,
    AX_NPU_CLK_ID       = 2,
    AX_ISP_CLK_ID       = 3,
    AX_MM_CLK_ID        = 4,
    AX_VPU_CLK_ID       = 5,
    AX_SYS_CLK_MAX_ID   = 6,
} AX_SYS_CLK_ID_E;

typedef struct axMOD_INFO_T {
    AX_MOD_ID_E enModId;
    AX_S32 s32GrpId;
    AX_S32 s32ChnId;
} AX_MOD_INFO_T;

typedef struct axLINK_DEST_S{
    AX_U32 u32DestNum;
    AX_MOD_INFO_T astDestMod[AX_LINK_DEST_MAXNUM];
} AX_LINK_DEST_T;

typedef enum {
    AX_MEMORY_SOURCE_CMM  = 0,
    AX_MEMORY_SOURCE_POOL = 1,
    AX_MEMORY_SOURCE_OS   = 2,
    AX_MEMORY_SOURCE_BUTT,
} AX_MEMORY_SOURCE_E;

typedef struct {
    AX_U64 u64PhyAddr;
    AX_VOID *pVirAddr;
} AX_MEMORY_ADDR_T;

typedef struct axOSD_BMP_ATTR_T {
    AX_U16 u16Alpha;
    AX_IMG_FORMAT_E enRgbFormat;
    AX_U8 *pBitmap;
    AX_U64 u64PhyAddr;
    AX_U32 u32BmpWidth;
    AX_U32 u32BmpHeight;
    AX_U32 u32DstXoffset;
    AX_U32 u32DstYoffset;
    AX_U32 u32Color;
    AX_BOOL bColorInv;
    AX_U32 u32ColorInv;
    AX_U32 u32ColorInvThr;
} AX_OSD_BMP_ATTR_T;

typedef enum {
    AX_ERR_INVALID_MODID        = 0x01,
    AX_ERR_INVALID_DEVID        = 0x02,
    AX_ERR_INVALID_GRPID        = 0x03,
    AX_ERR_INVALID_CHNID        = 0x04,
    AX_ERR_INVALID_PIPEID       = 0x05,
    AX_ERR_INVALID_STITCHGRPID  = 0x06,
    AX_ERR_ILLEGAL_PARAM        = 0x0A,
    AX_ERR_NULL_PTR             = 0x0B,
    AX_ERR_BAD_ADDR             = 0x0C,
    AX_ERR_SYS_NOTREADY         = 0x10,
    AX_ERR_BUSY                 = 0x11,
    AX_ERR_NOT_INIT             = 0x12,
    AX_ERR_NOT_CONFIG           = 0x13,
    AX_ERR_NOT_SUPPORT          = 0x14,
    AX_ERR_NOT_PERM             = 0x15,
    AX_ERR_EXIST                = 0x16,
    AX_ERR_UNEXIST              = 0x17,
    AX_ERR_NOMEM                = 0x18,
    AX_ERR_NOBUF                = 0x19,
    AX_ERR_NOT_MATCH            = 0x1A,
    AX_ERR_BUF_EMPTY            = 0x20,
    AX_ERR_BUF_FULL             = 0x21,
    AX_ERR_QUEUE_EMPTY          = 0x22,
    AX_ERR_QUEUE_FULL           = 0x23,
    AX_ERR_TIMED_OUT            = 0x27,
    AX_ERR_FLOW_END             = 0x28,
    AX_ERR_UNKNOWN              = 0x29,
    AX_ERR_OS_FAIL              = 0x30,
    AX_ERR_BUTT                 = 0x7F,
} AX_ERR_CODE_E;

#define AX_DEF_ERR( module, sub_module, errid) \
    ((AX_S32)( (0x80000000L) | ((module) << 16 ) | ((sub_module)<<8) | (errid) ))

typedef struct
{
    AX_S16 nX;
    AX_S16 nY;
} AX_POINT_T;

typedef struct {
    AX_BOOL    bEnable;
    AX_U32     nBgColor;
} AX_BGCOLOR_T;

typedef struct axCOLORKEY_T {
    AX_U16 u16Enable;
    AX_U16 u16Inv;
    AX_U32 u32KeyLow;
    AX_U32 u32KeyHigh;
} AX_COLORKEY_T;

typedef struct
{
    AX_U32   nColor;
    AX_BOOL  bColorInvEn;
    AX_U32   nColorInv;
    AX_U32   nColorInvThr;
} AX_BITCOLOR_T;

typedef struct {
    AX_BOOL             bEnable;
    AX_U16              nWidth;
    AX_U16              nHeight;
    AX_U32              nStride;
    AX_IMG_FORMAT_E     eFormat;
    AX_U64              u64PhyAddr[2];
    AX_FRAME_COMPRESS_INFO_T  stCompressInfo;
    AX_U8               nAlpha;
    AX_POINT_T          tOffset;
    AX_COLORKEY_T       tColorKey;
    AX_BITCOLOR_T       tBitColor;
} AX_OVERLAY_T;

typedef enum {
    AX_PYRA_MODE_GEN = 0,
    AX_PYRA_MODE_RCN = 1,
    AX_PYRA_MODE_BUTT,
} AX_PYRA_MODE_E;

typedef struct {
    AX_BOOL                     bEnable;
    AX_U16                      nWidth;
    AX_U16                      nHeight;
    AX_U32                      nStride;
    AX_IMG_FORMAT_E             eFormat;
    AX_U64                      nPhyAddr[2];
    AX_FRAME_COMPRESS_INFO_T    tCompressInfo;
    AX_U8                       PixelFormat;
    AX_BOOL                     bCropEnable;
    AX_S16                      nCropX0;
    AX_S16                      nCropY0;
    AX_U16                      nCropWidth;
    AX_U16                      nCropHeight;
} AX_PYRA_FRAME_T;

typedef enum _AX_VIN_COMB_MODE_E_ {
    AX_VIN_COMB_MODE_NONE = 0,
    AX_VIN_COMB_MODE0,
    AX_VIN_COMB_MODE_MAX
} AX_VIN_COMB_MODE_E;

#endif //_AX_GLOBAL_TYPE_H_
