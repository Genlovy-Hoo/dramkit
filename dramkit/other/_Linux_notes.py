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

# 后台运行不记录日志
nohup python xxx.py > /dev/null 2>&1 &
# windows后台挂起运行python脚本
start /b python xxx.py

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

# zip打包多个文件（夹）
zip -r xxx.zip path1 path2 ...

# 复制命令
cp [-adfilprsu] 源文件(source) 目标文件(destination)
cp [option] source1 source2 source3 ... directory
# 参数说明：
-a: 是指 archive 的意思，也说是指复制所有的目录
-d: 若源文件为连接文件 (link file)，则复制连接文件属性而非文件本身
-f: 强制 (force)，若有重复或其它疑问时，不会询问用户，而强制复制
-i: 若目标文件 (destination) 已存在，在覆盖时会先询问是否真的操作
-l: 建立硬连接 (hard link) 的连接文件，而非复制文件本身
-p: 与文件的属性一起复制，而非使用默认属性
-r: 递归复制，用于目录的复制操作
-s: 复制成符号连接文件 (symbolic link)，即 “快捷方式” 文件
-u: 若目标文件比源文件旧，更新目标文件 

# 移动命令
mv [-fiv] source destination
# 参数说明：
-f: force，强制直接移动而不询问
-i: 若目标文件 (destination) 已经存在，就会询问是否覆盖
-u: 若目标文件已经存在，且源文件比较新，才会更新

# 删除命令
rm [fir] 文件或目录
# 参数说明：
-f: 强制删除
-i: 交互模式，在删除前询问用户是否操作
-r: 递归删除，常用在目录的删除

# ubuntu修改用户
passwd [user]

# 查看端口占用情况
netstat -anp # 或
netstat -tln

# 查看目前所有应用端口
netstat -nplt

# 查看某个端口占用情况
netstat -tunlp | grep 端口号

# 修改远程登录端口
vim /etc/ssh/sshd_config # 配置中修改端口
service sshd restart # 重启sshd服务

# linux在线下载anaconda3（改版本）
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh

# 将anaconda替换为默认python环境
vim /etc/profile # 在末尾添加：
export PATH=/root/anaconda3/bin:$PATH
source /etc/profile # 使之生效

# 创建快捷方式
ln -s 目标路径 快捷方式名称

# 查看当前目录下所有文件列表
ls -a
ll -h
ll -ht # 按时间降序排列
ll -htr # 按时间升序排列
ll -hS # 按大小降序排列
ll -hSr # 按大小升序排列

# ubuntu安装node.js
# 参考https://blog.csdn.net/wxtcstt/article/details/128800620
sudo apt-get remove nodejs
sudo apt-get remove npm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.2/install.sh | bash
chmod +x ~/.nvm/nvm.sh
source ~/.bashrc 
nvm -v
nvm install 18
node -v
npm -v

npm install -g pnpm # npm安装pnpm

# unzip解压覆盖
unzip -o xxx.zip

# '''