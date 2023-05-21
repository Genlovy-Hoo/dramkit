# -*- coding: utf-8 -*-

# Python笔记

# 参考：
# pip命令：https://blog.csdn.net/fancunshuai/article/details/124994040

py_notes = \
r'''
# conda查看镜像
conda config --show-sources

# conda设置镜像
# 首先用命令创建配置文件（或手动创建）.condarc：
conda config --set show_channel_urls yes # 设置搜索时显示通道地址
# 在.condarc中添加常用镜像源：
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
# conda清除索引缓存使新配置生效
conda clean -i
# conda命令行添加镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# pip查看镜像源
pip config list

# pip安装时带临时镜像
pip install pkgname -i https://mirror.baidu.com/pypi/simple
pip install pkgname -i https://pypi.org/simple # PYPI官方镜像

# pip设置镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 在pip.ini（windows）中设置镜像
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = https://pypi.tuna.tsinghua.edu.cn

# pip常用镜像列表
https://pypi.tuna.tsinghua.edu.cn/simple # 清华
https://pypi.mirrors.ustc.edu.cn/simple # 中科大
https://pypi.hustunique.com/ # 华中科大
https://pypi.hustunique.com/ #  山东理工
https://mirror.baidu.com/pypi/simple/ # 百度
https://mirrors.aliyun.com/pypi/simple/ # 阿里
http://mirrors.sohu.com/Python/ # 搜狐
https//pypi.doubanio.com/simple # 豆瓣

# conda查看已有环境
conda env list
conda info -e

# conda创建虚拟环境
conda create -n envname python=x.x

# conda激活虚拟环境
conda activate envname

# conda退出虚拟环境
conda deactivate

# conda虚拟环境安装spyder
conda install spyder

# conda虚拟环境安装notebook
conda install nb_conda

# pip安装忽略依赖
--no-dependencies
 
# pip强制重新安装所有软件包，即使它们已经是最新的
--force-reinstall

# pip批量安装文件夹下所有的whl包
pip install /dirpath/*.whl

# '''