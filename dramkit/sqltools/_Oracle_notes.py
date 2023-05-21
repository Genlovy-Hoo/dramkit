# -*- coding: utf-8 -*-

# Oracle笔记

orcl_notes = \
r'''
-- 基于OraDB19Home1社区版本
-- 大写字母为SQL命令，小写字母为对象名称
-- []中为可选参数

-- 登录: sqlplus 输入用户名(sys as sysdba)和密码

CONN / as sysdba; /*sys连接，出现`SP2-0640: 未连接`错误时用*/

-- 查看数据库版本信息
SELECT * FROM v$version;

SHOW con_name; /*显示数据库名称*/
SHOW USER; /*查看当期用户*/

SHOW PDBS; /*查看可插拔数据库*/
ALTER SESSION SET CONTAINER = ORCLPDB; /*切换到可插拔数据库*/
ALTER SESSION SET CONTAINER = CDB$ROOT; /*切换到CDB容器*/

-- 查看用户信息
SELECT username, user_id, account_status FROM dba_users [where username like '%HYY%'];
DESC dba_users; /*查看dba_users表结构信息*/
SELECT * FROM all_users; /*Oracle的所有用户信息*/
-- 查询SCOTT用户的信息
SELECT USER_ID, USERNAME, ACCOUNT_STATUS, EXPIRY_DATE, DEFAULT_TABLESPACE FROM DBA_USERS WHERE USERNAME='SCOTT';
SELECT * FROM user_users; /*当前用户信息*/
-- 查询当前用户的信息
SELECT USER_ID, USERNAME, EXPIRY_DATE, DEFAULT_TABLESPACE FROM USER_USERS;
SELECT * FROM dba_users; /*查询所有用户的信息，能够查询的信息比dba_users要少*/

-- 创建用户（在CBD容器中要在用户名前加C##）
CREATE USER C##username IDENTIFIED BY password [DEFAULT TABLESPACE "tablespace_name"];
GRANT CONNECT, RESOURCE, DBA TO C##username; /*连接授权*/
CONN C##username/password; /*连接到新用户*/

-- 删除用户
-- 查看用户连接情况
SELECT username, sid, serial# FROM v$session where username = 'username';
-- 找到要删除用户的sid和serial#，并删除
ALTER SYSTEM KILL SESSION '133,5873'; /*133,5873对应sid, serial#*/
-- 删除用户
DROP USER username CASCADE;

-- sys as sysdba登录之后创建scott用户
-- https://blog.csdn.net/weixin_57263771/article/details/128270914
-- https://blog.csdn.net/qq_45678901/article/details/124440133
-- 先切换到pdb
ALTER SESSION SET CONTAINER = ORCLPDB;
-- 再创建SCOTT用户
CREATE USER SCOTT IDENTIFIED BY TIGER;
GRANT CONNECT, RESOURCE, DBA TO SCOTT; /*连接授权*/
-- 若出现错误：ORA-01109: 数据库未打开，则先使用startup打开数据库
startup;
-- 修改ORACLE_HOME下的network/admin/tnsnames.ora文件，添加如下内容：
ORCLPDB = 
  (DESCRIPTION = 
    (ADDRESS = (PROTOCOL = TCP)(HOST = localhost)(PORT = 1521)) 
    (CONNECT_DATA = 
      (SERVER = DEDICATED) 
      (SERVICE_NAME = orclpdb) 
    ) 
  )
-- 修改ORACLE_HOME下的rdbms/admin/utlsampl.sql：
CONNECT SCOTT/TIGER改为CONNECT SCOTT/TIGER@ORCLPDB (tiger小写的地方也改为大写)
-- 运行修改后的utlsampl.sql
-- @D:\BaiduNetdiskDownload\WINDOWS.X64_193000_db_home\rdbms\admin\utlsampl.sql
@$ORACLE_HOME/rdbms/admin/utlsampl.sql
-- 重新连接并切换到pdb
CONN / as sysdba;
ALTER SESSION SET CONTAINER = ORCLPDB;
-- 解锁scott用户（老版oracle自带scott用户，需要解锁，新版scott用户要手动建，不存在解锁）
ALTER USER SCOTT ACCOUNT UNLOCK;
-- 测试
SELECT * FROM EMP;

-- 所有触发器
SELECT trigger_name FROM all_triggers;
-- 用户触发器
SELECT * FROM user_triggers;
-- 所有存储过程
SELECT * FROM user_procedures;
-- 所有视图
SELECT * FROM user_views;
-- 所有表
SELECT * FROM user_tables;

-- 查看表空间使用情况
SELECT a.tablespace_name "表空间名",
       a.bytes / 1024 / 1024 "表空间大小(M)",
       (a.bytes - b.bytes) / 1024 / 1024 "已使用空间(M)",
       b.bytes / 1024 / 1024 "空闲空间(M)",
       round(((a.bytes - b.bytes) / a.bytes) * 100, 2) "使用比"
FROM
    (SELECT tablespace_name, sum(bytes) bytes
     FROM dba_data_files
     GROUP BY tablespace_name) a,
    (SELECT tablespace_name, sum(bytes) bytes, max(bytes) largest
     FROM dba_free_space
     GROUP BY tablespace_name) b
WHERE a.tablespace_name = b.tablespace_name
ORDER BY ((a.bytes - b.bytes) / a.bytes) DESC;
-- 查看表空间情况
SELECT tablespace_name, file_name, autoextensible FROM dba_data_files [where tablespace_name = '表空间名称'];
SELECT file_id, file_name, tablespace_name, autoextensible, increment_by FROM dba_data_files [WHERE tablespace_name = '表空间名称'] ORDER BY file_id desc;
-- 查看当前用户表空间名称
SELECT username, default_tablespace FROM user_users;
-- 开启自动扩展表空间
ALTER DATABASE DATAFILE '对应的数据文件路径' AUTOEXTEND ON;
-- 关闭自动扩展表空间
ALTER DATABASE DATAFILE '对应的数据文件路径' AUTOEXTEND OFF;
-- 修改表空间大小
ALTER DATABASE DATAFILE '对应的数据文件路径' RESIZE 2000M;
-- 为表空间新增一个数据文件
ALTER TABLESPACE tablespace_name ADD DATAFILE '新增的数据文件路径' SIZE  1G AUTOEXTEND ON NEXT 200M MAXSIZE unlimited;

-- 创建表空间(初始大小5M，自动增长每次增长5M，最大占用3000M)
-- 注：表空间名要用双引号？
CREATE TABLESPACE "表空间名字" DATAFILE '数据文件路径.dbf'
SIZE 5M AUTOEXTEND ON next 5M
MAXSIZE 3000M;

-- 修改用户默认关联表空间
ALTER USER user_name DEFAULT TABLESPACE tablespace_name;

-- 清理表
-- 查询对应表空间下最占空间的20个对象
SELECT OWNER, SEGMENT_NAME, SEGMENT_TYPE, total||'M'
FROM
    (SELECT OWNER, SEGMENT_NAME, SEGMENT_TYPE, bytes/1024/1024 total
     FROM dba_segments
     WHERE TABLESPACE_NAME = 'tablespace_name'
     order by bytes/1024/1024 desc)
WHERE ROWNUM < 21;
-- 删除表（带PURGE不放入回收站）
DROP TABLE tablename [PURGE];
-- 查看回收站
SELECT OWNER, OBJECT_NAME, ts_name FROM dba_recyclebin;
-- 清空回收站数据
PURGE recyclebin;
-- 清空回收站所有数据，需要sys登录
PURGE DBA_RECYCLEBIN;
-- 删除表空间数据文件（须先删除里面的表，否则报错ORA-03262）
ALTER TABLESPACE tablespace_name DROP DATAFILE '数据文件路径';

-- 获取创建表空间的语句
SELECT dbms_metadata.get_ddl('TABLESPACE', 'TABLESPACE_NAME要大写') FROM dual;
SELECT dbms_metadata.get_ddl('TABLESPACE', 'USERS') FROM dual;

-- 查询ORACLE数据块大小
SHOW PARAMETER db_block_size;

-- 查询SID
SELECT instance_name FROM V$INSTANCE;

-- 创建表
CREATE TABLE table_name(var_name1 var_type1, var_name2 var_type2, ...); /*创建表*/
-- 查询表创建语句
SELECT dbms_metadata.get_ddl('TABLE', 'TABLENAME要大写') FROM dual;
SELECT dbms_metadata.get_ddl('TABLE', 'TEST1') FROM dual;

-- 赋予select any dictionary权限
GRANT SELECT ANY DICTIONARY TO username;

-- 查询表字段信息
DESC table_name; /*或*/
SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, NULLABLE
FROM user_tab_columns
WHERE table_name = UPPER('TABLE_NAME大写');

# '''


