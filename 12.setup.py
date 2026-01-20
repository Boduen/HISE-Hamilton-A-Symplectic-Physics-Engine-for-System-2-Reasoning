from setuptools import setup, find_packages
import torch
import os


# Check for CUDA availability to install Physics Kernels
cuda_available = torch.cuda.is_available()


# Dependency list
requirements = [
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "triton>=2.1.0; platform_system=='Linux'", # Triton physics engine
    "einops>=0.7.0",
    "scipy>=1.10.0",
    "matplotlib",
    "streamlit",
    "pandas",
    "seaborn"
]


setup(
    name="hise-pro",
    version="1.0.0-alpha",
    author="Boduen Wang",
    description="Holographic Inertial Syntax Engine (HISE) with System 1/2 Dynamics",
    long_description=open("README.md", "r", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/boduen-wang/HISE-Pro",
    packages=find_packages(exclude=["tests", "csrc", "visualization"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    # In a real scenario, we would add CUDAExtension here for csrc/fused_soft_tcm.cu
    # But for this version, we rely on the JIT-compiled Triton kernels in hise/kernels/
)