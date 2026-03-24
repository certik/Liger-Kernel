from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "numpy._core",
            sources=["src/numpy/_core.c"],
            extra_compile_args=["-O2", "-std=c11"],
        ),
    ],
)
