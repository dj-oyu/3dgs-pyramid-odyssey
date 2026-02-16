/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Shanghai) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Shanghai) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Shanghai) Co., Ltd.
 *
 **************************************************************************************************/

#ifndef _AX_IVPS_TYPE_H_
#define _AX_IVPS_TYPE_H_
#include "ax_global_type.h"

#define IVPS_SUCC 0x00

#define AX_SUB_ID_IVPS         0x01
#define AX_SUB_ID_RGN          0x02
#define AX_SUB_ID_GDC          0x03

#define AX_ERROR_IVPS(e) AX_DEF_ERR(AX_ID_IVPS, AX_SUB_ID_IVPS, (e))
#define AX_ERROR_RGN(e)  AX_DEF_ERR(AX_ID_IVPS, AX_SUB_ID_RGN, (e))
#define AX_ERROR_GDC(e)  AX_DEF_ERR(AX_ID_IVPS, AX_SUB_ID_GDC, (e))


/*******************************IVPS ERROR CODE************************************/
#define AX_ERR_IVPS_INVALID_MODID                 AX_ERROR_IVPS(AX_ERR_INVALID_MODID)
#define AX_ERR_IVPS_INVALID_DEVID                 AX_ERROR_IVPS(AX_ERR_INVALID_DEVID)
#define AX_ERR_IVPS_INVALID_CHNID                 AX_ERROR_IVPS(AX_ERR_INVALID_CHNID)
#define AX_ERR_IVPS_ILLEGAL_PARAM                 AX_ERROR_IVPS(AX_ERR_ILLEGAL_PARAM)
#define AX_ERR_IVPS_NULL_PTR                      AX_ERROR_IVPS(AX_ERR_NULL_PTR)
#define AX_ERR_IVPS_BAD_ADDR                      AX_ERROR_IVPS(AX_ERR_BAD_ADDR)
#define AX_ERR_IVPS_SYS_NOTREADY                  AX_ERROR_IVPS(AX_ERR_SYS_NOTREADY)
#define AX_ERR_IVPS_BUSY                          AX_ERROR_IVPS(AX_ERR_BUSY)
#define AX_ERR_IVPS_NOT_INIT                      AX_ERROR_IVPS(AX_ERR_NOT_INIT)
#define AX_ERR_IVPS_NOT_CONFIG                    AX_ERROR_IVPS(AX_ERR_NOT_CONFIG)
#define AX_ERR_IVPS_NOT_SUPPORT                   AX_ERROR_IVPS(AX_ERR_NOT_SUPPORT)
#define AX_ERR_IVPS_NOT_PERM                      AX_ERROR_IVPS(AX_ERR_NOT_PERM)
#define AX_ERR_IVPS_EXIST                         AX_ERROR_IVPS(AX_ERR_EXIST)
#define AX_ERR_IVPS_UNEXIST                       AX_ERROR_IVPS(AX_ERR_UNEXIST)
#define AX_ERR_IVPS_NOMEM                         AX_ERROR_IVPS(AX_ERR_NOMEM)
#define AX_ERR_IVPS_NOBUF                         AX_ERROR_IVPS(AX_ERR_NOBUF)
#define AX_ERR_IVPS_BUF_EMPTY                     AX_ERROR_IVPS(AX_ERR_BUF_EMPTY)
#define AX_ERR_IVPS_BUF_FULL                      AX_ERROR_IVPS(AX_ERR_BUF_FULL)
#define AX_ERR_IVPS_QUEUE_EMPTY                   AX_ERROR_IVPS(AX_ERR_QUEUE_EMPTY)
#define AX_ERR_IVPS_QUEUE_FULL                    AX_ERROR_IVPS(AX_ERR_QUEUE_FULL)
#define AX_ERR_IVPS_TIMED_OUT                     AX_ERROR_IVPS(AX_ERR_TIMED_OUT)
#define AX_ERR_IVPS_FLOW_END                      AX_ERROR_IVPS(AX_ERR_FLOW_END)
#define AX_ERR_IVPS_UNKNOWN                       AX_ERROR_IVPS(AX_ERR_UNKNOWN)

/*******************************IVPS RGN ERROR CODE*********************************/
#define AX_ERR_IVPS_RGN_INVALID_GRPID             AX_ERROR_RGN(AX_ERR_INVALID_GRPID)
#define AX_ERR_IVPS_RGN_BUSY                      AX_ERROR_RGN(AX_ERR_BUSY)
#define AX_ERR_IVPS_RGN_UNEXIST                   AX_ERROR_RGN(AX_ERR_UNEXIST)
#define AX_ERR_IVPS_RGN_ILLEGAL_PARAM             AX_ERROR_RGN(AX_ERR_ILLEGAL_PARAM)
#define AX_ERR_IVPS_RGN_NOBUF                     AX_ERROR_RGN(AX_ERR_NOBUF)

#define AX_IVPS_MIN_IMAGE_WIDTH 32
#define AX_IVPS_MAX_IMAGE_WIDTH 8192

#define AX_IVPS_MIN_IMAGE_HEIGHT 32
#define AX_IVPS_MAX_IMAGE_HEIGHT 8192

#define AX_IVPS_MAX_GRP_NUM 256
#define AX_IVPS_MAX_OUTCHN_NUM 5
#define AX_IVPS_MAX_FILTER_NUM_PER_OUTCHN 2
#define AX_IVPS_INVALID_FRMRATE (-1)

typedef AX_S32 IVPS_GRP;
typedef AX_S32 IVPS_CHN;
typedef AX_S32 IVPS_FILTER;

typedef AX_S32 IVPS_RGN_GRP;
typedef AX_U32 IVPS_RGB;

typedef enum
{
    AX_IVPS_CHN_FLIP_NONE = 0,
    AX_IVPS_CHN_FLIP,
    AX_IVPS_CHN_MIRROR,
    AX_IVPS_CHN_FLIP_AND_MIRROR,
    AX_IVPS_CHN_FLIP_BUTT
} AX_IVPS_CHN_FLIP_MODE_E;

typedef struct
{
    AX_S16 nX;
    AX_S16 nY;
    AX_U16 nW;
    AX_U16 nH;
} AX_IVPS_RECT_T;

typedef struct
{
    AX_S16 nX;
    AX_S16 nY;
} AX_IVPS_POINT_T;

typedef struct
{
    AX_F32 fX;
    AX_F32 fY;
} AX_IVPS_POINT_NICE_T;

typedef struct
{
    AX_U16 nW;
    AX_U16 nH;
} AX_IVPS_SIZE_T;

typedef enum
{
    AX_IVPS_MOSAIC_BLK_SIZE_2 = 0,
    AX_IVPS_MOSAIC_BLK_SIZE_4,
    AX_IVPS_MOSAIC_BLK_SIZE_8,
    AX_IVPS_MOSAIC_BLK_SIZE_16,
    AX_IVPS_MOSAIC_BLK_SIZE_32,
    AX_IVPS_MOSAIC_BLK_SIZE_64,
    AX_IVPS_MOSAIC_BLK_SIZE_BUTT
} AX_IVPS_MOSAIC_BLK_SIZE_E;

typedef enum
{
    AX_IVPS_ROTATION_0   = 0,
    AX_IVPS_ROTATION_90  = 1,
    AX_IVPS_ROTATION_180 = 2,
    AX_IVPS_ROTATION_270 = 3,
    AX_IVPS_ROTATION_BUTT
} AX_IVPS_ROTATION_E;

typedef enum
{
    AX_IVPS_ASPECT_RATIO_STRETCH = 0,
    AX_IVPS_ASPECT_RATIO_AUTO = 1,
    AX_IVPS_ASPECT_RATIO_MANUAL = 2,
    AX_IVPS_ASPECT_RATIO_BUTT
} AX_IVPS_ASPECT_RATIO_E;

typedef enum
{
    AX_IVPS_ASPECT_RATIO_HORIZONTAL_CENTER = 0,
    AX_IVPS_ASPECT_RATIO_HORIZONTAL_LEFT = 1,
    AX_IVPS_ASPECT_RATIO_HORIZONTAL_RIGHT = 2,
    AX_IVPS_ASPECT_RATIO_VERTICAL_CENTER = AX_IVPS_ASPECT_RATIO_HORIZONTAL_CENTER,
    AX_IVPS_ASPECT_RATIO_VERTICAL_TOP = AX_IVPS_ASPECT_RATIO_HORIZONTAL_LEFT,
    AX_IVPS_ASPECT_RATIO_VERTICAL_BOTTOM = AX_IVPS_ASPECT_RATIO_HORIZONTAL_RIGHT,
} AX_IVPS_ASPECT_RATIO_ALIGN_E;

/***************************************************************************************************************/
/*                                                   GDC                                                       */
/***************************************************************************************************************/
#define AX_IVPS_GDC_MAX_IMAGE_WIDTH  8192
#define AX_IVPS_GDC_MAX_IMAGE_HEIGHT 8192
#define AX_IVPS_GDC_MIN_IMAGE_WIDTH  256
#define AX_IVPS_GDC_MIN_IMAGE_HEIGHT 256
#define AX_IVPS_FISHEYE_MAX_RGN_NUM  9
#define AX_IVPS_GDC_MAX_HANDLE_NUM  32

typedef AX_S32 GDC_HANDLE;

typedef enum
{
    AX_IVPS_GDC_BYPASS = 0,
    AX_IVPS_GDC_FISHEYE,
    AX_IVPS_GDC_MAP_USER,
    AX_IVPS_GDC_BUTT
} AX_IVPS_GDC_TYPE_E;

typedef enum
{
    AX_IVPS_FISHEYE_MOUNT_MODE_DESKTOP = 0,
    AX_IVPS_FISHEYE_MOUNT_MODE_CEILING = 1,
    AX_IVPS_FISHEYE_MOUNT_MODE_WALL = 2,
    AX_IVPS_FISHEYE_MOUNT_MODE_BUTT
} AX_IVPS_FISHEYE_MOUNT_MODE_E;

typedef enum
{
    AX_IVPS_FISHEYE_VIEW_MODE_PANORAMA = 0,
    AX_IVPS_FISHEYE_VIEW_MODE_NORMAL = 1,
    AX_IVPS_FISHEYE_VIEW_MODE_BYPASS = 2,
    AX_IVPS_FISHEYE_VIEW_MODE_BUTT
} AX_IVPS_FISHEYE_VIEW_MODE_E;

typedef struct
{
    AX_IVPS_FISHEYE_VIEW_MODE_E eViewMode;
    AX_U16 nInRadius;
    AX_U16 nOutRadius;
    AX_U16 nPan;
    AX_U16 nTilt;
    AX_U16 nCenterX;
    AX_U16 nCenterY;
    AX_U16 nHorZoom;
    AX_U16 nVerZoom;
    AX_IVPS_RECT_T tOutRect;
} AX_IVPS_FISHEYE_RGN_ATTR_T;

typedef struct
{
    AX_BOOL bBgColor;
    AX_U32 nBgColor;
    AX_S16 nHorOffset;
    AX_S16 nVerOffset;
    AX_U8 nTrapezoidCoef;
    AX_S16 nFanStrength;
    AX_IVPS_FISHEYE_MOUNT_MODE_E eMountMode;
    AX_U8 nRgnNum;
    AX_BOOL bRgnUpdate;
    AX_U8 nRgnUpdateIdx;
    AX_BOOL bRoiXY;
    AX_IVPS_FISHEYE_RGN_ATTR_T tFisheyeRgnAttr[AX_IVPS_FISHEYE_MAX_RGN_NUM];
} AX_IVPS_FISHEYE_ATTR_T;

typedef struct
{
    AX_U16 nMeshStartX;
    AX_U16 nMeshStartY;
    AX_U16 nMeshWidth;
    AX_U16 nMeshHeight;
    AX_U8 nMeshNumH;
    AX_U8 nMeshNumV;
    AX_S32 *pUserMap;
    AX_U64 nMeshTablePhyAddr;
} AX_IVPS_MAP_USER_ATTR_T;

typedef struct
{
    AX_IVPS_GDC_TYPE_E eGdcType;
    union {
        AX_IVPS_FISHEYE_ATTR_T tFisheyeAttr;
        AX_IVPS_MAP_USER_ATTR_T tMapUserAttr;
    };
    AX_U16 nSrcWidth;
    AX_U16 nSrcHeight;
    AX_U16 nDstStride;
    AX_U16 nDstWidth;
    AX_U16 nDstHeight;
    AX_IMG_FORMAT_E eDstFormat;
} AX_IVPS_GDC_ATTR_T;


typedef enum
{
    AX_IVPS_DEWARP_BYPASS = 0,
    AX_IVPS_DEWARP_MAP_USER,
    AX_IVPS_DEWARP_PERSPECTIVE,
    AX_IVPS_DEWARP_LDC,
    AX_IVPS_DEWARP_LDC_V2,
    AX_IVPS_DEWARP_LDC_PERSPECTIVE,
    AX_IVPS_DEWARP_BUTT
} AX_IVPS_DEWARP_TYPE_E;

typedef struct
{
    AX_S64 nMatrix[9];
} AX_IVPS_PERSPECTIVE_ATTR_T;

typedef struct
{
    AX_BOOL bAspect;
    AX_S16 nXRatio;
    AX_S16 nYRatio;
    AX_S16 nXYRatio;
    AX_S16 nCenterXOffset;
    AX_S16 nCenterYOffset;
    AX_S16 nDistortionRatio;
    AX_S8 nSpreadCoef;
} AX_IVPS_LDC_ATTR_T;

typedef struct
{
    AX_U32 nXFocus;
    AX_U32 nYFocus;
    AX_U32 nXCenter;
    AX_U32 nYCenter;
    AX_S64 nDistortionCoeff[8];
} AX_IVPS_LDC_V2_ATTR_T;

typedef struct
{
    AX_IVPS_DEWARP_TYPE_E eDewarpType;

    AX_IVPS_ROTATION_E eRotation;
    AX_BOOL bMirror;
    AX_BOOL bFlip;
    union
    {
        AX_IVPS_MAP_USER_ATTR_T tMapUserAttr;
        AX_IVPS_LDC_ATTR_T tLdcAttr;
        AX_IVPS_LDC_V2_ATTR_T tLdcV2Attr;
    };
    AX_IVPS_PERSPECTIVE_ATTR_T tPerspectiveAttr;
} AX_IVPS_GDC_CFG_T;

typedef struct
{
    AX_BOOL bCrop;
    AX_IVPS_RECT_T tCropRect;
    AX_U16 nDstWidth;
    AX_U16 nDstHeight;
    AX_U32 nDstStride;
    AX_IMG_FORMAT_E eImgFormat;

    AX_BOOL bPerspective;
    AX_IVPS_PERSPECTIVE_ATTR_T tPerspectiveAttr;
} AX_IVPS_DEWARP_ATTR_T;

typedef struct
{
      AX_BOOL           bAlphaEnable;
      AX_BOOL           bAlphaReverse;
      AX_U64            u64PhyAddr;
} AX_IVPS_ALPHA_LUT_T;

typedef struct
{
    AX_POOL_SOURCE_E ePoolSrc;
    AX_U8 nFrmBufNum;
    AX_POOL PoolId;
} AX_IVPS_POOL_ATTR_T;

typedef enum {
    AX_COORD_ABS = 0,
    AX_COORD_RATIO,
    AX_COORD_BUTT
} AX_COORD_E;

typedef struct {
    AX_BOOL bEnable;
    AX_COORD_E eCoordMode;
    AX_IVPS_RECT_T tCropRect;
} AX_IVPS_CROP_INFO_T;

typedef struct
{
    AX_BOOL bEnable;
    AX_U32 nHorLength;
    AX_U32 nVerLength;
} AX_IVPS_CORNER_RECT_ATTR_T;

typedef enum
{
    AX_IVPS_SCALE_RANGE_0 = 0,
    AX_IVPS_SCALE_RANGE_1,
    AX_IVPS_SCALE_RANGE_2,
    AX_IVPS_SCALE_RANGE_3,
    AX_IVPS_SCALE_RANGE_4,
    AX_IVPS_SCALE_RANGE_5,
    AX_IVPS_SCALE_RANGE_6,
    AX_IVPS_SCALE_RANGE_7,
    AX_IVPS_SCALE_RANGE_8,
    AX_IVPS_SCALE_RANGE_BUTT
} AX_IVPS_SCALE_RANGE_TYPE_E;

typedef struct {
    AX_IVPS_SCALE_RANGE_TYPE_E eHorScaleRange;
    AX_IVPS_SCALE_RANGE_TYPE_E eVerScaleRange;
} AX_IVPS_SCALE_RANGE_T;

typedef enum
{
    AX_IVPS_COEF_LEVEL_0 = 0,
    AX_IVPS_COEF_LEVEL_1,
    AX_IVPS_COEF_LEVEL_2,
    AX_IVPS_COEF_LEVEL_3,
    AX_IVPS_COEF_LEVEL_4,
    AX_IVPS_COEF_LEVEL_5,
    AX_IVPS_COEF_LEVEL_6,
    AX_IVPS_COEF_LEVEL_BUTT,
} AX_IVPS_COEF_LEVEL_E;


typedef struct {
    AX_IVPS_COEF_LEVEL_E eHorLuma;
    AX_IVPS_COEF_LEVEL_E eHorChroma;
    AX_IVPS_COEF_LEVEL_E eVerLuma;
    AX_IVPS_COEF_LEVEL_E eVerChroma;
} AX_IVPS_SCALE_COEF_LEVEL_T;

#endif
