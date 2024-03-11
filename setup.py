from setuptools import setup, find_packages

setup(
    name="module_logging",
    version="1.0.0",
    author="Eric.Wang",
    author_email="wangfangfei@baidu.com",
    description="logging on moudle and aten op level",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    entry_points={
        'console_scripts': [
            'module_logging = module_logging.__main__:main'
        ]
    }
    # py_modules=["module_logging"],
)
