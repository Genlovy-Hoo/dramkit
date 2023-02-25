# -*- coding: utf-8 -*-

# Linux笔记

linux_notes = \
r'''
# 参考
https://blog.csdn.net/changlina_1989/article/details/111144018

# 命令行运行python并将输出保存日志
# 日志覆盖：
python -u xxx.py > xxx.log 2>&1 &
# 日志追加：
python -u xxx.py >> xxx.log 2>&1 &

du -h #  查看当前文件夹及子文件夹大小信息
du -sh #  查看当前文件夹大小信息
du -h * # 查看当前文件夹下所有文件大小信息
# 查看磁盘情况
df -h

# xshell上传文件命令
rz
# xshell下载文件命令
sz filepath

# 查找文件
find dirpath -name filename
find / -name "*site.xml" # 在根目录查找后缀为site.xml的文件

# centos安装与gcc对应版本的g++
yum install gcc-c++
gcc -v # 查看gcc版本
g++ -v # 查看g++版本

# 查看当前所在目录的绝对路经
pwd

# 查看命令历史记录
history

# 查看docker容器列表
docker ps

# 进入docker容器
docker exec -it docker_id /bin/bash # 最后一个是docker ps结果的COMMAND那一列

# '''