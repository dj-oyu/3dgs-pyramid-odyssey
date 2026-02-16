/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Shanghai) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Shanghai) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Shanghai) Co., Ltd.
 *
 **************************************************************************************************/

#ifndef _AX_IVE_API_H_
#define _AX_IVE_API_H_

#include "ax_ive_type.h"

#ifdef __cplusplus
extern "C"
{
#endif

AX_S32 AX_IVE_Init(AX_VOID);
AX_VOID AX_IVE_Exit(AX_VOID);

AX_S32 AX_IVE_Query(AX_IVE_HANDLE IveHandle, AX_BOOL *pbFinish, AX_BOOL bBlock);

AX_S32 AX_IVE_DMA(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_DATA_T *pstSrc, AX_IVE_DST_DATA_T *pstDst,
    AX_IVE_DMA_CTRL_T *pstDmaCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_Add(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc1, AX_IVE_SRC_IMAGE_T *pstSrc2,
    AX_IVE_DST_IMAGE_T *pstDst, AX_IVE_ADD_CTRL_T *pstAddCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_Sub(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc1, AX_IVE_SRC_IMAGE_T *pstSrc2,
    AX_IVE_DST_IMAGE_T *pstDst, AX_IVE_SUB_CTRL_T *pstSubCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_And(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc1, AX_IVE_SRC_IMAGE_T *pstSrc2,
    AX_IVE_DST_IMAGE_T *pstDst, AX_BOOL bInstant);

AX_S32 AX_IVE_Or(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc1, AX_IVE_SRC_IMAGE_T *pstSrc2,
    AX_IVE_DST_IMAGE_T *pstDst, AX_BOOL bInstant);

AX_S32 AX_IVE_Xor(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc1, AX_IVE_SRC_IMAGE_T *pstSrc2,
    AX_IVE_DST_IMAGE_T *pstDst, AX_BOOL bInstant);

AX_S32 AX_IVE_Mse(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc1, AX_IVE_SRC_IMAGE_T *pstSrc2,
    AX_IVE_DST_IMAGE_T *pstDst, AX_IVE_MSE_CTRL_T *pstMseCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_CannyHysEdge(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc1, AX_IVE_SRC_IMAGE_T *pstSrc2,
    AX_IVE_DST_IMAGE_T *pstDst, AX_IVE_HYS_EDGE_CTRL_T *pstCannyHysEdgeCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_CannyEdge(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstDst,
    AX_IVE_CANNY_EDGE_CTRL_T *pstCannyEdgeCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_CCL(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstDst,
    AX_IVE_DST_MEM_INFO_T *pstBlob, AX_IVE_CCL_CTRL_T *pstCclCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_Erode(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstDst,
    AX_IVE_ERODE_CTRL_T *pstErodeCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_Dilate(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstDst,
    AX_IVE_DILATE_CTRL_T *pstDilateCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_Filter(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstDst,
    AX_IVE_FILTER_CTRL_T *pstFltCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_Hist(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_MEM_INFO_T *pstDst,
    AX_BOOL bInstant);

AX_S32 AX_IVE_EqualizeHist(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_MEM_INFO_T *pstDst,
    AX_IVE_EQUALIZE_HIST_CTRL_T *pstEqualizeHistCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_Integ(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstDst,
    AX_IVE_INTEG_CTRL_T *pstIntegCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_MagAndAng(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc1, AX_IVE_SRC_IMAGE_T *pstSrc2,
    AX_IVE_DST_IMAGE_T *pstDstMag, AX_IVE_DST_IMAGE_T *pstDstAng, AX_BOOL bInstant);

AX_S32 AX_IVE_Sobel(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstDst,
    AX_IVE_SOBEL_CTRL_T *pstSobelCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_GMM(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstFg,
    AX_IVE_DST_IMAGE_T *pstBg, AX_IVE_MEM_INFO_T *pstModel, AX_IVE_GMM_CTRL_T *pstGmmCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_GMM2(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstFg,
    AX_IVE_DST_IMAGE_T *pstBg, AX_IVE_MEM_INFO_T *pstModel, AX_IVE_GMM2_CTRL_T *pstGmm2Ctrl, AX_BOOL bInstant);

AX_S32 AX_IVE_Thresh(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstDst,
    AX_IVE_THRESH_CTRL_T *pstThrCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_16BitTo8Bit(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstDst,
    AX_IVE_16BIT_TO_8BIT_CTRL_T *pst16BitTo8BitCtrl, AX_BOOL bInstant);

AX_S32 AX_IVE_CropImage(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pastDst[],
    AX_IVE_RECT_U16_T *pastSrcBoxs[], AX_IVE_CROP_IMAGE_CTRL_T *pstCropImageCtrl, AX_IVE_ENGINE_E enEngine, AX_BOOL bInstant);

AX_S32 AX_IVE_CropResize(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pastDst[],
    AX_IVE_RECT_U16_T *pastSrcBoxs[], AX_IVE_CROP_RESIZE_CTRL_T *pstCropResizeCtrl, AX_IVE_ENGINE_E enEngine, AX_BOOL bInstant);

AX_S32 AX_IVE_CropResizeForSplitYUV(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc1, AX_IVE_SRC_IMAGE_T *pstSrc2,
    AX_IVE_DST_IMAGE_T *pastDst1[], AX_IVE_DST_IMAGE_T *pastDst2[], AX_IVE_RECT_U16_T *pastSrcBoxs[],
    AX_IVE_CROP_RESIZE_CTRL_T *pstCropResizeCtrl, AX_IVE_ENGINE_E enEngine, AX_BOOL bInstant);

AX_S32 AX_IVE_CSC(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_DST_IMAGE_T *pstDst,
    AX_IVE_ENGINE_E enEngine, AX_BOOL bInstant);

AX_S32 AX_IVE_CropResize2(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc, AX_IVE_IMAGE_T *pastDst[], AX_IVE_RECT_U16_T *pastSrcBoxs[],
    AX_IVE_RECT_U16_T *pastDstBoxs[], AX_IVE_CROP_IMAGE_CTRL_T *pstCropResize2Ctrl, AX_IVE_ENGINE_E enEngine, AX_BOOL bInstant);

AX_S32 AX_IVE_CropResize2ForSplitYUV(AX_IVE_HANDLE *pIveHandle, AX_IVE_SRC_IMAGE_T *pstSrc1, AX_IVE_SRC_IMAGE_T *pstSrc2,
    AX_IVE_IMAGE_T *pastDst1[], AX_IVE_IMAGE_T *pastDst2[], AX_IVE_RECT_U16_T *pastSrcBoxs[], AX_IVE_RECT_U16_T *pastDstBoxs[],
    AX_IVE_CROP_IMAGE_CTRL_T *pstCropResize2Ctrl, AX_IVE_ENGINE_E enEngine, AX_BOOL bInstant);

AX_S32 AX_IVE_MAU_MatMul(AX_IVE_HANDLE *pIveHandle, AX_IVE_MAU_MATMUL_INPUT_T *pstSrc, AX_IVE_MAU_MATMUL_OUTPUT_T *pstDst,
    AX_IVE_MAU_MATMUL_CTRL_T *pstMatMulCtrl, AX_IVE_ENGINE_E enEngine, AX_BOOL bInstant);

AX_S32 AX_IVE_NPU_CreateMatMulHandle(AX_IVE_MATMUL_HANDLE *pHandle, AX_IVE_NPU_MATMUL_CTRL_T *pstMatMulCtrl);
AX_S32 AX_IVE_NPU_DestroyMatMulHandle(AX_IVE_MATMUL_HANDLE *pHandle);

AX_S32 AX_IVE_NPU_MatMul(AX_IVE_MATMUL_HANDLE hHandle, AX_IVE_MAU_MATMUL_INPUT_T *pstSrc, AX_IVE_MAU_MATMUL_OUTPUT_T *pstDst,
    AX_IVE_ENGINE_E enEngine, AX_BOOL bInstant);

#ifdef __cplusplus
}
#endif

#endif //_AX_IVE_API_H_
