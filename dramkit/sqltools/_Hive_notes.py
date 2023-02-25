# -*- coding: utf-8 -*-

# Hive笔记

hive_notes = \
r'''
/* 
MERGE INTO报错：
FAILED: Execution Error, return code 2 from org.apache.hadoop.hive.ql.exec.mr.MapredLocalTask
解决办法：
SET hive.auto.convert.join=false;
*/

/*
INSERT INTO报错:
FAILED: Execution Error, return code 1 from org.apache.hadoop.hive.ql.exec.StatsTask
解决办法1：
先TRUNCATE TABLE再INSERT INTO
原因不知道，参考：https://blog.csdn.net/xingchensuiyue/article/details/121949045?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-4-121949045-blog-111553256.pc_relevant_multi_platform_whitelistv3&spm=1001.2101.3001.4242.3&utm_relevant_index=7
解决办法2：
set hive.stats.autogather=false
参考：https://www.jianshu.com/p/bebcceac24e7
*/

/*
update和delete报错：
FAILED: SemanticException [Error 10294]: Attempt to do update or delete using transaction manager that does not support these operations.
解决方案：
修改hive-site.xml如下配置，并在建表时指定STORED AS ORC TBLPROPERTIES('transactional'='true')
<property>
    <name>hive.support.concurrency</name>
    <value>true</value>
</property>
<property>
    <name>hive.enforce.bucketing</name>
    <value>true</value>
</property>
<property>
    <name>hive.exec.dynamic.partition.mode</name>
    <value>nonstrict</value>
</property>
<property>
    <name>hive.txn.manager</name>
    <value>org.apache.hadoop.hive.ql.lockmgr.DbTxnManager</value>
</property>
<property>
    <name>hive.compactor.initiator.on</name>
    <value>true</value>
</property>
<property>
    <name>hive.compactor.worker.threads</name>
    <value>1</value>
</property>
<property>
    <name>hive.in.test</name>
    <value>true</value>
</property>
参考：https://my.oschina.net/xiaominmin/blog/5527662
*/

# 查看是否开启hadoop安全模式
hadoop dfsadmin -safemode get

# 关闭hadoop安全模式
hadoop dfsadmin -safemode leave

# 打开hadoop安全模式
hadoop dfsadmin -safemode enter

# 报错：
This command is not allowed on an ACID table xxx.xxx_table  with a non-ACID transaction manager
解决方案：
SET hive.support.concurrency=true;
SET hive.txn.manager=org.apache.hadoop.hive.ql.lockmgr.DbTxnManager;

# 客户端kerberos认证命令（先cd到kerberos目录，windows一般为program Files\MIT\Kerberos\bin）
# TODO: 命令具体意义
kinit -kt hive.keytab hive/hadoop-003-250.guoyuanml.com@GUOYUANML.COM

# '''