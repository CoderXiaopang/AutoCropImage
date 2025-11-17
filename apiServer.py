# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import random
import numpy as np
from autocrop import cropper
import base64
from typing import Optional, List
import traceback
import os
import glob

app = FastAPI(title="图像智能裁剪服务")

# 全局初始化 AutoCropper
autocropper = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化 AutoCropper"""
    global autocropper
    try:
        def _find_local_weight(model_name: str) -> str:
            torch_home = os.environ.get("TORCH_HOME", "/root/.cache/torch/hub")
            ckpt_dir = os.path.join(torch_home, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            pattern = "mobilenet_*.pth" if model_name == "mobilenetv2" else "shufflenet_*.pth"
            matches = glob.glob(os.path.join(ckpt_dir, pattern))
            return matches[0] if matches else ""

        # 裁剪模型：优先本地权重
        local_model_path = _find_local_weight("mobilenetv2")
        # 人脸模型：仅当本地存在时才启用人脸过滤，避免在线下载失败
        torch_home = os.environ.get("TORCH_HOME", "/root/.cache/torch/hub")
        dsfd_local = ""
        for candidate in [
            os.path.join(torch_home, "checkpoints", "WIDERFace_DSFD_RES152.pth"),
            os.path.join(torch_home, "hub", "checkpoints", "WIDERFace_DSFD_RES152.pth"),
        ]:
            if os.path.isfile(candidate):
                dsfd_local = candidate
                break
        use_face = True if dsfd_local else False
        autocropper = cropper.AutoCropper(
            model='mobilenetv2',
            cuda=True,
            model_path=local_model_path,
            use_face_detector=use_face
        )
        print("AutoCropper 初始化成功")
    except Exception as e:
        print(f"AutoCropper 初始化失败: {e}")
        autocropper = None


class CropRequest(BaseModel):
    image_base64: str
    ratio: str  # 格式: "宽:高" (如 "16:9", "1:1") 或 "circular"


class CropResponse(BaseModel):
    status: str  # "success" 或 "error"
    cropped_image_base64: str
    original_image_with_box_base64: str
    bounding_box: List[int]  # [x1, y1, x2, y2]
    message: str = "裁剪成功"


def apply_circular_mask(image):
    """将正方形图像裁剪成圆形，背景变为透明"""
    h, w, _ = image.shape
    if h != w:
        side = min(h, w)
        y_start = (h - side) // 2
        x_start = (w - side) // 2
        image = image[y_start:y_start + side, x_start:x_start + side]
        h, w = side, side
    
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(center)
    cv2.circle(mask, center, radius, 255, -1)
    
    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_rgba[:, :, 3] = mask
    return image_rgba


def base64_to_image(base64_str: str):
    """将 base64 字符串转换为 OpenCV 图像"""
    try:
        # 移除可能的 data URL 前缀
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # 移除可能的空格和换行符
        base64_str = base64_str.strip().replace('\n', '').replace('\r', '')
        
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("cv2.imdecode 返回 None")
        
        return img
    except Exception as e:
        raise ValueError(f"无法解码 base64 图像: {e}")


def image_to_base64(img, is_png=False):
    """将 OpenCV 图像转换为 base64 字符串"""
    try:
        ext = '.png' if is_png else '.jpg'
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if not is_png else []
        
        success, buffer = cv2.imencode(ext, img, encode_param)
        if not success:
            raise ValueError("cv2.imencode 失败")
        
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        raise ValueError(f"无法编码图像为 base64: {e}")


@app.post("/crop", response_model=CropResponse)
async def crop_image(request: CropRequest):
    """
    智能裁剪图像接口
    
    参数:
    - image_base64: 原始图像的 base64 编码
    - ratio: 裁剪比例，格式为 "宽:高" (如 "16:9", "1:1") 或 "circular" (圆形裁剪)
    
    返回:
    - status: 处理状态 ("success" 或 "error")
    - cropped_image_base64: 裁剪后的图像 base64
    - original_image_with_box_base64: 带裁剪框的原图 base64
    - bounding_box: 裁剪区域坐标 [x1, y1, x2, y2]
    - message: 处理消息
    """
    try:
        print(f"收到裁剪请求，比例: {request.ratio}")
        
        if autocropper is None:
            raise HTTPException(status_code=500, detail="AutoCropper 未初始化")
        
        # 解析图像
        print("正在解析 base64 图像...")
        img = base64_to_image(request.image_base64)
        print(f"图像解析成功，尺寸: {img.shape}")
        
        # 解析裁剪比例
        is_circular = False
        if request.ratio == 'circular':
            CROP_WIDTH, CROP_HEIGHT = 1, 1
            is_circular = True
            print("使用圆形裁剪模式")
        else:
            try:
                w_str, h_str = request.ratio.split(':')
                CROP_WIDTH, CROP_HEIGHT = float(w_str), float(h_str)
                print(f"裁剪比例: {CROP_WIDTH}:{CROP_HEIGHT}")
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"无效的比例格式 '{request.ratio}'，请使用 '宽:高' 格式或 'circular'"
                )
        
        # 转换为 RGB 用于裁剪
        img_rgb = img[:, :, (2, 1, 0)]
        
        # 执行智能裁剪
        print("正在执行智能裁剪...")
        crop_ret = autocropper.crop(
            img_rgb, 
            topK=1, 
            crop_height=CROP_HEIGHT, 
            crop_width=CROP_WIDTH,
            filter_face=True, 
            single_face_center=True
        )
        
        if not crop_ret or len(crop_ret) == 0:
            print("未找到合适的裁剪区域")
            raise HTTPException(status_code=404, detail="未找到合适的裁剪区域")
        
        # 获取第一个裁剪区域
        bbox = crop_ret[0]
        print(f"裁剪框: {bbox}")
        
        cropped_image = img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :]
        print(f"裁剪后图像尺寸: {cropped_image.shape}")
        
        # 如果是圆形裁剪，应用圆形蒙版
        is_png = False
        if is_circular:
            print("应用圆形蒙版...")
            cropped_image = apply_circular_mask(cropped_image)
            is_png = True
        
        # 生成带裁剪框的原图
        print("生成带裁剪框的原图...")
        img_with_boxes = img.copy()
        r, g, b = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
        cv2.rectangle(img_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (b, g, r), 3)
        
        # 转换为 base64
        print("转换图像为 base64...")
        cropped_base64 = image_to_base64(cropped_image, is_png)
        original_with_box_base64 = image_to_base64(img_with_boxes)
        
        print(f"裁剪成功！裁剪图像 base64 长度: {len(cropped_base64)}")
        
        return CropResponse(
            status="success",
            cropped_image_base64=cropped_base64,
            original_image_with_box_base64=original_with_box_base64,
            bounding_box=[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            message=f"裁剪成功，区域: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"处理图像时出错: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/")
async def root():
    """健康检查接口"""
    return {
        "message": "图像智能裁剪服务运行中",
        "status": "healthy",
        "autocropper_initialized": autocropper is not None
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok" if autocropper is not None else "error"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)