# -*- coding: utf-8 -*-
import os
import cv2
import random
import argparse
import numpy as np
from autocrop import cropper
import time


# --- 辅助函数：用于圆形裁剪 (无变动) ---
def apply_circular_mask(image):
    """
    将一个正方形图像裁剪成圆形，背景变为透明。
    """
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


# --- 主处理函数 (无变动，其内部逻辑已支持任意比例) ---
def process_images_in_folder(input_dir, output_dir, ratio_choice):
    """
    根据指定的比例，批量处理文件夹中的所有图片，并保存裁剪结果。
    """
    # 将比例中的':'替换为'_'用作文件夹名，例如'16:9' -> '16_9'
    subfolder_name = ratio_choice.replace(':', '_')
    final_output_dir = os.path.join(output_dir, subfolder_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"输出文件夹 '{final_output_dir}' 已准备就绪。")

    is_circular = False
    # 优先处理特殊的 'circular' 选项
    if ratio_choice == 'circular':
        CROP_WIDTH, CROP_HEIGHT = 1, 1  # 圆形裁剪基于1:1的正方形
        is_circular = True
        print("已选择圆形裁剪模式。")
    else:
        # 尝试解析 '宽:高' 格式的字符串
        try:
            w_str, h_str = ratio_choice.split(':')
            # CROP_WIDTH, CROP_HEIGHT = int(w_str), int(h_str)
            CROP_WIDTH, CROP_HEIGHT = float(w_str), float(h_str)
            print(f"已选择裁剪比例: {CROP_WIDTH}:{CROP_HEIGHT}")
        except ValueError:
            print(f"错误：无效的比例格式 '{ratio_choice}'。请使用 '宽:高' 的格式 (例如 '16:9') 或 'circular'。")
            return

    # 初始化计时
    init_start_time = time.time()
    try:
        autocropper = cropper.AutoCropper(model='mobilenetv2', cuda=True, use_face_detector=True)
        init_end_time = time.time()
        print(f"AutoCropper 初始化成功，耗时 {init_end_time - init_start_time:.2f} 秒。")
    except Exception as e:
        print(f"AutoCropper 初始化失败: {e}")
        return

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    total_images = len(image_files)
    print(f"在 '{input_dir}' 找到 {total_images} 张图片，开始处理...")

    # 为总体计时和计数做准备
    processed_count = 0
    failed_count = 0
    overall_start_time = time.time()

    for filename in image_files:
        image_start_time = time.time()
        input_path = os.path.join(input_dir, filename)
        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"警告：无法读取图片 '{filename}'，已跳过。")
                failed_count += 1
                continue

            print(f"正在处理: {filename}")
            img_rgb = img[:, :, (2, 1, 0)]

            crop_ret = autocropper.crop(img_rgb, topK=1, crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH,
                                        filter_face=True, single_face_center=True)
            if not crop_ret:
                print(f"在 '{filename}' 中未找到合适的裁剪区域。")
                failed_count += 1
                continue

            base_name, orig_ext = os.path.splitext(filename)
            for idx, bbox in enumerate(crop_ret):
                cropped_image = img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :]

                output_ext = orig_ext
                if is_circular:
                    cropped_image = apply_circular_mask(cropped_image)
                    output_ext = '.png'  # 圆形带透明通道，强制为png

                output_crop_name = f"{base_name}_crop_{idx}{output_ext}"
                output_crop_path = os.path.join(final_output_dir, output_crop_name)
                cv2.imwrite(output_crop_path, cropped_image)

            # (可选) 保存带裁剪框的原图
            img_with_boxes = img.copy()
            for bbox in crop_ret:
                r, g, b = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
                cv2.rectangle(img_with_boxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (b, g, r), 2)
            output_boxes_name = f"{base_name}_with_boxes{orig_ext}"
            output_boxes_path = os.path.join(final_output_dir, output_boxes_name)
            cv2.imwrite(output_boxes_path, img_with_boxes)

            processed_count += 1
            image_end_time = time.time()
            image_duration = image_end_time - image_start_time
            print(f"处理完成: '{filename}'，耗时 {image_duration:.2f} 秒。")

        except Exception as e:
            print(f"处理图片 '{filename}' 时发生错误: {e}")
            failed_count += 1

    # 计算并打印总体报告
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    print("\n" + "=" * 50)
    print(" 全 部 任 务 处 理 完 成 ".center(50, "="))
    print("=" * 50)
    print(f"总计图片数: {total_images}")
    print(f"成功处理数: {processed_count}")
    print(f"失败/跳过数: {failed_count}")
    print(f"总耗时: {total_duration:.2f} 秒")
    if processed_count > 0:
        # 这里修正为用总处理时间除以成功处理的图片数，得到更准确的平均时间
        avg_time = total_duration / processed_count
        print(f"平均每张成功处理耗时: {avg_time:.2f} 秒")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用指定的宽高比批量裁剪文件夹中的图片。")
    parser.add_argument('-i', '--input', type=str, default='imgs/裁剪测试图',
                        help="存放所有待处理图片的输入文件夹路径。")
    parser.add_argument('-o', '--output', type=str, default='/root/autodl-tmp/result_crop/test/',
                        help="存放所有处理结果的根输出文件夹路径。")

    # --- 主要修改点在这里 ---
    parser.add_argument('-r', '--ratio', type=str, default='1:2.39',
                        # 移除了 choices=[...] 参数
                        help="指定裁剪比例 (格式 '宽:高'，如 '16:9', '2:3') 或进行圆形裁剪 ('circular')。")
    # -------------------------

    args = parser.parse_args()
    process_images_in_folder(args.input, args.output, args.ratio)