# -*- coding: utf-8 -*-

# Git笔记

'''
# gitGUI连接github Repository步骤

step1：git命令：ssh-keygen -t rsa -C "邮箱" ，生成SSH Key，默认存放在用户文件夹的.ssh下面

step2：将生成的SSH Key添加到github的SSH Key设置里面。即复制.ssh/id_rsa.pub里面的内容添加到github设置里面的SSH Key中（这是本地与github的通信密码）。

step3：用git命令：ssh -T git@github.com进行连接测试

step4：用git命令添加用户信息：

​		git config --global user.name github用户名

​		git config --global user.email github邮箱

这样，本地与github的连接设置就完成了。

step5：在github上新建Repostitory

step6：打开git GUI将github上的项目克隆到本地

step7：在git GUI里面进行文件修改，提交，上传等操作

# '''