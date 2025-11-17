# -*- coding: utf-8 -*-
"""
@File       : cropImage.py
@Author     : Duangang Qu
@Email      : quduangang@outlook.com
@Created    : 2025/11/6 13:59
@Modified   : 2025/11/7 11:55 (由 Gemini 重构)
@Software   : PyCharm
@Description: 封装 API 调用为函数，接受 Base64 并返回结果。

"""
import requests
import base64
import os
import time
import sys


# --- 辅助函数 (保持不变) ---

def image_to_base64(image_path: str) -> str | None:
    """将图片文件转换为 Base64 字符串"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误：测试图片 '{image_path}' 不存在。")
        return None
    except Exception as e:
        print(f"读取文件 {image_path} 失败: {e}")
        return None


def base64_to_image_file(b64_string: str, output_path: str):
    """将 Base64 字符串解码并保存为图片文件"""
    try:
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        img_data = base64.b64decode(b64_string)
        with open(output_path, "wb") as f:
            f.write(img_data)
    except base64.binascii.Error as e:
        print(f"Base64 解码失败 {output_path}: {e}")
    except Exception as e:
        print(f"保存文件 {output_path} 失败: {e}")


# --- 主调用函数 (已重构) ---

def get_all_crop_ratios(b64_image_string: str,ratio ) -> dict:
    """
    接受 Base64 图像，调用 API 测试所有比例，并返回所有结果。

    Args:
        b64_image_string: Base64 编码的源图像。

    Returns:
        一个字典，键是比例 (e.g., '16:9')，值是包含结果的子字典。
        e.g., {'16:9': {'cropped_image_base64': '...', 'original_image_with_box_base64': '...'}}
    """
    API_URL = "https://u284779-9d2a-faab6091.westx.seetacloud.com:8443/crop"


    # 用于存储所有返回结果的字典
    all_results = {}


    print(f"\n--- 正在请求裁剪比例: {ratio} ---")
    # 1. 准备 JSON 载荷
    payload = {
        "image_base64": b64_image_string,
        "ratio": ratio
    }
    start_time = time.time()
    try:
        # 2. 发送 POST 请求
        response = requests.post(API_URL, json=payload, timeout=60.0)
        duration = time.time() - start_time
        print(f"请求耗时: {duration:.2f} 秒")
        # 3. 处理响应
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("裁剪成功！")
                print(f"  > 裁剪框 (bbox): {data.get('bounding_box')}")
                # 存储结果，而不是保存文件
                all_results[ratio] = {
                    "cropped_image_base64": data.get("cropped_image_base64"),
                    "original_image_with_box_base64": data.get("original_image_with_box_base64"),
                    "bounding_box": data.get('bounding_box')
                }
            else:
                print(f"API 返回成功状态码，但内容指示错误: {data.get('message')}")
        else:
            # 处理 HTTP 错误
            print(f"API 请求失败，状态码: {response.status_code}")
            try:
                error_data = response.json()
                print(f"  > 错误详情: {error_data.get('detail')}")
            except requests.exceptions.JSONDecodeError:
                print(f"  > 无法解析错误响应: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"请求连接失败: {e}")

    return all_results


if __name__ == "__main__":


    TEST_IMAGE_PATH = "/Users/quxiaopang/Desktop/beauty_crop/imgs/裁剪测试图/jimeng-2025-08-20-8254-foreign youngsters are having party, kee....png"
    OUTPUT_DIR = "api_crop_results"

    # 检查路径是否有效
    if TEST_IMAGE_PATH == "/Users/quxiaopang/Desktop/unnamed.jpg" and not os.path.exists(TEST_IMAGE_PATH):
        print(f"错误：默认测试图片 '{TEST_IMAGE_PATH}' 不存在。")
        print("请在 'TEST_IMAGE_PATH' 变量中指定一张有效的图片路径后再运行。")
        sys.exit(1)

    # 2. 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. 将图片编码为 Base64
    print(f"正在读取并编码图片: {TEST_IMAGE_PATH}")
    b64_image = image_to_base64(TEST_IMAGE_PATH)

    if b64_image:
        # 4. 调用函数并传入 Base64
        print(f"--- 开始调用 API ---")
        ratios_to_test = '1:1'
        crop_results = get_all_crop_ratios(b64_image,ratios_to_test)

        # 5. 处理函数返回的结果 (保存到文件)
        if crop_results:
            print(f"\n--- API 调用完成，正在保存 {len(crop_results)} 组结果 ---")

            for ratio, data in crop_results.items():
                print(f"正在保存 {ratio}...")

                safe_ratio = ratio.replace(':', '_')
                ext = '.png' if ratio == 'circular' else '.jpg'

                # 保存裁剪后的图片
                if data.get("cropped_image_base64"):
                    crop_filename = f"result_crop_{safe_ratio}{ext}"
                    crop_path = os.path.join(OUTPUT_DIR, crop_filename)
                    base64_to_image_file(data["cropped_image_base64"], crop_path)
                    print(f"  > 裁剪图已保存至: {crop_path}")

                # 保存带框的原图
                if data.get("original_image_with_box_base64"):
                    debug_filename = f"result_debug_{safe_ratio}.jpg"
                    debug_path = os.path.join(OUTPUT_DIR, debug_filename)
                    base64_to_image_file(data["original_image_with_box_base64"], debug_path)
                    print(f"  > 调试图已保存至: {debug_path}")

            print("\n--- 所有文件保存完毕 ---")

        else:
            print("--- API 调用完成，但未返回任何有效结果 ---")
    else:
        print("--- 因图片读取失败，程序中止 ---")