from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fast_hadamard_transform_op",
    ext_modules=[
        CUDAExtension(
            name="fast_hadamard_transform_op",
            sources=[
                "binding.hip",
                "fast_hadamard_transform.hip"
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": []  # ROCm ignore
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
