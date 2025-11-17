import setuptools
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# 获取包含此 setup.py 文件的目录的绝对路径
current_dir = os.path.dirname(os.path.realpath(__file__))


def generate_extensions():
    """
    为项目生成 C++/CUDA 扩展。
    它会自动检测 CUDA 是否可用，并构建相应的版本。
    """
    # 定义相对于项目根目录的源文件路径
    rod_src_path = os.path.join('autocrop', 'model', 'rod_align', 'src')
    roi_src_path = os.path.join('autocrop', 'model', 'roi_align', 'src')

    # 获取 PyTorch 的默认包含路径。如果 CUDA 可用，它会自动感知。
    torch_includes = torch.utils.cpp_extension.include_paths(cuda=torch.cuda.is_available())

    # 两个扩展通用的包含目录
    include_dirs = [
                       current_dir,
                       os.path.join(current_dir, rod_src_path),
                       os.path.join(current_dir, roi_src_path)
                   ] + torch_includes

    extensions = []

    if torch.cuda.is_available():
        print("✅ CUDA 可用。正在编译 CUDA 扩展。")
        # 定义 ROI Align 的 CUDA 扩展
        roi_align_cuda = CUDAExtension(
            name='roi_align_api',
            sources=[
                os.path.join(roi_src_path, 'roi_align_cuda.cpp'),
                os.path.join(roi_src_path, 'roi_align_kernel.cu')
            ],
            include_dirs=include_dirs
        )
        # 定义 ROD Align 的 CUDA 扩展
        rod_align_cuda = CUDAExtension(
            name='rod_align_api',
            sources=[
                os.path.join(rod_src_path, 'rod_align_cuda.cpp'),
                os.path.join(rod_src_path, 'rod_align_kernel.cu')
            ],
            include_dirs=include_dirs
        )
        extensions.extend([roi_align_cuda, rod_align_cuda])
    else:
        print("⚠️ CUDA 不可用。正在编译仅支持 CPU 的 C++ 扩展。")
        # 定义 ROI Align 的 C++ 扩展
        roi_align_cpp = CppExtension(
            name='roi_align_api',
            sources=[os.path.join(roi_src_path, 'roi_align.cpp')],
            include_dirs=include_dirs
        )
        # 定义 ROD Align 的 C++ 扩展
        rod_align_cpp = CppExtension(
            name='rod_align_api',
            sources=[os.path.join(rod_src_path, 'rod_align.cpp')],
            include_dirs=include_dirs
        )
        extensions.extend([roi_align_cpp, rod_align_cpp])

    return extensions


setuptools.setup(
    name='auto_crop',
    version='0.3.0',
    description='Smart auto cropping tool that supports any aspect ratio',
    long_description=open('README.md', 'r', encoding='utf-8').read(),  # 添加了 encoding='utf-8' 以确保兼容性
    long_description_content_type='text/markdown',
    url='https://github.com/lih627/autocrop',
    python_requires='>=3.8',  # 更新以适应现代 Python 版本
    license='MIT',

    # 动态生成需要编译的扩展列表
    ext_modules=generate_extensions(),

    # 这是告诉 setuptools 使用 PyTorch 自定义构建器的关键命令
    cmdclass={
        'build_ext': BuildExtension
    },

    # 让 pip 正确处理包的安装
    install_requires=[
        'torch>=2.0',  # 更新为更现代的基础版本
        'torchvision>=0.15.0',
        'numpy',
        'opencv-python',
        'face-detection>=0.1.4'
    ],

    packages=setuptools.find_packages()
)