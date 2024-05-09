import setuptools


setuptools.setup(
    name='waifu-sensor',
    version='2.0.0',
    author='RimoChan',
    author_email='the@librian.net',
    description='waifu-sensor',
    long_description=open('readme.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RimoChan/waifu-sensor',
    packages=['waifu_sensor'],
    package_data={'waifu_sensor': ['人均值.json.xz', '人均值v2.json.xz']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'huggingface-hub~=0.15.1',
        'numpy~=1.24.3',
        'hbutils~=0.9.1',
        'Pillow~=10.0.1',
    ],
    python_requires='>=3.9',
)
