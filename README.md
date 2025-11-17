# AutoCropImage

美学自动裁切，基于 PyTorch 与人脸检测，支持任意比例裁切。

## 环境要求

- Python `3.12`（Ubuntu 22.04）
- PyTorch `2.5.1`（CUDA `12.4`）
- GPU 可选；无 GPU 时自动编译 CPU 版本的 C++ 扩展

## 安装

- 安装 PyTorch（CUDA 12.4）：
  - `pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124`
  - CPU 仅需：`pip install torch==2.5.1 torchvision==0.20.1`
- 编译并安装本项目：
  - `python setup.py install`
- 额外依赖：
  - `pip install opencv-python`

提示：`setup.py` 会根据 `torch.cuda.is_available()` 自动编译 CUDA/CPU 扩展，源码位于 `autocrop/model/roi_align` 与 `autocrop/model/rod_align`。

## 模型准备（必须）

- 将以下权重文件移动到 ` /root/.cache/torch/hub/checkpoints `：
  - `WIDERFace_DSFD_RES152.pth`（人脸检测模型）
  - `mobilenet_0.625_... .pth`（裁切评分模型）
  - `shufflenet_0.615_... .pth`（裁切评分模型）
- 示例命令：
  - `sudo mkdir -p /root/.cache/torch/hub/checkpoints`
  - `sudo cp WIDERFace_DSFD_RES152.pth /root/.cache/torch/hub/checkpoints/`
  - `sudo cp mobilenet_*.pth /root/.cache/torch/hub/checkpoints/`
  - `sudo cp shufflenet_*.pth /root/.cache/torch/hub/checkpoints/`

说明：裁剪模型通过 `torch.hub.load_state_dict_from_url` 自动使用该缓存目录（见 `autocrop/model/cropping_model.py:117-131`）；提前放置权重可避免在线下载与网络依赖。

## 快速上手

- Python 调用：
  - `from autocrop import cropper`
  - `autocropper = cropper.AutoCropper(model='mobilenetv2', cuda=True, use_face_detector=True)`
  - 读入 `BGR` 图并转 `RGB` 后，调用 `autocropper.crop(...)` 即可返回 `[xmin, ymin, xmax, ymax]`

- API 服务：
  - `python apiServer.py`
  - POST `/crop`，参数：`image_base64`、`ratio`（如 `16:9`、`1:1` 或 `circular`）

示例接口实现参考 `apiServer.py:105-200`；裁剪核心逻辑见 `autocrop/cropper.py:124-169`。

## 依赖与编译说明

- 运行依赖（见 `setup.py:90-96` 与 `environment.yml`）：
  - `torch`、`torchvision`、`numpy`、`opencv-python`、`face-detection`
  - API：`fastapi`、`uvicorn`、`pydantic`
- 编译要求：
  - CPU：需要 C++ 编译环境（gcc/clang）
  - GPU：需要 CUDA 12.4（`nvcc`、`CUDA_HOME`），自动构建 `roi_align_api` 与 `rod_align_api`

## 目录结构

- `autocrop/` 核心库与模型、算子实现
- `apiServer.py` FastAPI 服务端示例
- `anyratio.py`、`cropImage.py` 简单脚本示例
- `setup.py` 构建与依赖声明

## 许可

- 参考与复现自 GAIC 与 DSFD，上游许可请见其仓库。



