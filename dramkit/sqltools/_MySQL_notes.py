# -*- coding: utf-8 -*-

# MySQL笔记

mysql_notes = \
r'''
-- 基于MySQL Server 5.7社区版本
-- 大写字母为SQL命令，小写字母为对象名称
-- SQL语句正常以“;”结束，若以“\G”结束则可以显示更美观的结果
-- []中为可选参数

-- 登录
mysql -h ip地址 -P 端口 -u 用户名 -p

-- 查看版本
SELECT VERSION()

-- 写入文件
查询语句 INTO OUTFILE 'xxx.csv'

-- 导出表结构和数据
mysqldump [--no-defaults(报--no-beep错误时使用)] -uxxx -pxxx [导出选项] dbname [tbname] > filepath.sql /*常用导出选项：-d只导出表结构，-t只导出表数据，不填则导出结构和数据*/
mysqldump [--no-defaults(报--no-beep错误时使用)] -uxxx -p [导出选项] dbname [tbname] > filepath.sql /*密码独立输入*/

-- 导入
mysql -uxxx -p dbname < filepath.sql
source filepath.sql /*登录mysql之后use指定数据库之后执行*/

-- 改密码（注意：-uroot和-p旧密码之间没有空格）
mysqladmin -uroot -p旧密码 password 新密码 /*mysqladmin改密*/

-- mysql5.7改密
SET PASSWORD FOR "username"@"host" = PASSWORD("newpassword");
SET PASSWORD = PASSWORD("newpassword"); /*当前已登录用户改密*/

-- mysql8改密
SET PASSWORD FOR "username"@"host" = "newpassword";
SET PASSWORD = "newpassword"; /*当前已登录用户改密*/

-- 查看binlog是否开启
SHOW VARIABLES LIKE "log_bin";

-- 查看数据库日志相关变量
SHOW VARIABLES LIKE "log_%";

-- 查看日志文件列表
SHOW BINARY LOGS;

-- 查看最新日志文件名称
SHOW MASTER STATUS;

-- 产生一个新的日志文件
FLUSH LOGS;

-- 查看日志内容
SHOW BINLOG EVENTS IN 'xxx.xxxxxx'; /*xxx.xxxxxx是日志文件名称*/

-- 查看数据文件目录
SHOW VARIABLES LIKE "datadir";
SELECT @@datadir;

-- 查看安装文件目录
SHOW VARIABLES LIKE "basedir";
SELECT @@basedir;

-- （windows）mysql57开启bin log，在配置文件my.ini中[mysqld]下添加内容：log-bin=mysql-bin
-- （windows）mysql8关闭bin log，在配置文件my.ini中[mysqld]下添加内容：skip-log-bin

-- 2006 MySQL server has gone解决
show global status like 'uptime'; /*查看mysql的运行时长*/
show global variables like '%timeout'; /*查看超时设置*/
set global wait_timeout=3600*48; /*设置连接超时*/
show global status like 'com_kill'; /*查看连接进程被kill*/
show global variables like 'max_allowed_packet'; /*查看查询结果大小限制*/
set global max_allowed_packet=1024*1024*16; /*修改查询结果大小限制*/

-- Ubuntu设置MySql局域网可以访问
-- 1. 编辑 /etc/mysql/mysql.conf.d/mysqld.cnf
-- 把里面的bind-address = 127.0.0.1
-- 改成  # bind-address = 127.0.0.1 进行屏蔽
-- 2：
mysql -uroot -ppassword /*登录*/
use mysql;
update user set host = '%' where user ='root';
grant all privileges on *.* to 'root'@'%' with grant option;
flush privileges;
exit;
-- 3. 重启mysql
sudo /etc/init.d/mysql restart

-- 刚安装没有密码时设置初始密码
[sudo] mysql /*登入数据库*/
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password by 'mynewpassword'; /*登入数据库之后*/
[sudo] mysql_secure_installation /*退出数据库之后，出现的问题都选n*/

-- 创建新用户
CREATE USER "username"@"host" IDENTIFIED BY "password";
CREATE USER "ff"@"localhost" IDENTIFIED BY "xxxxxxxxxxx"; /*创建用户ff，允许本地登录*/
CREATE USER "ff"@"192.168.0.102" IDENTIFIED BY "xxxxxxxxxxx"; /*创建用户ff，允许指定ip登录*/
CREATE USER "ff"@"192.168.0.%" IDENTIFIED BY "xxxxxxxxxxx"; /*创建用户ff，允许指定ip范围登录*/
CREATE USER "ff"@"%" IDENTIFIED BY "xxxxxxxxxxx"; /*创建用户ff，允许任意ip登录*/
CREATE USER "ff"@"%"; /*创建用户ff，允许任意ip免密登录*/

-- 用户赋权
GRANT privileges ON dbname.tbname TO "username"@"host";
GRANT ALL ON db1.* TO "ff"@"%"; /*赋予用户ff以db1库的所有权限*/
GRANT SELECT ON db1.* TO "ff"@"%"; /*赋予用户ff以db1库所有表的SELECT权限*/
GRANT SELECT, INSERT, UPDATE ON db1.tb2 TO "ff"@"%"; /*赋予用户ff以db1库tb1表的SELECT, INSERT, UPDATE权限*/

-- 取消权限
REVOKE privileges ON dbname.tbname FROM "username"@"host";
REVOKE INSERT, UPDATE ON test.* FROM 'ff'@'%'; /*取消ff用户在test库的INSERT, UPDATE权限*/
REVOKE ALL ON test.* FROM 'ff'@'%'; /*取消ff用户在test库的所有权限*/

-- 权限刷新
FLUSH PRIVILEGES;

SHOW GRANTS FOR "ff"@"%"; /*查看权限*/

-- 删除用户
DROP USER "username"@"host"; 
DROP USER "ff"@"localhost";

STATUS; /*查看当前MySQL状态，其中Current database一行会显示当前数据库*/

CREATE DATABASE database_name; /*创建数据库*/
SHOW DATABASES; /*查看现有的数据库名称*/
SELECT DATABASE() \G /*查看当前数据库*/
USE database_name; /*选择数据库*/
DROP DATABASE database_name; /*删除数据库*/

SHOW ENGINES;
SHOW ENGINES \G /*查看支持的存储引擎*/
SHOW VARIABLES LIKE 'have%'; /*查看支持的存储引擎*/
SHOW VARIABLES LIKE '%storage_engine%'; /*查看默认存储引擎*/

HELP CONTENTS; /*+查看帮助目录*/
HELP DATA TYPES; /*查看数据类型帮助*/
HELP INT; /*查看整数类型帮助*/

CREATE TABLE table_name(var_name1 var_type1, var_name2 var_type2, ...); /*创建表*/

-- 查看数据库创建语句
SHOW CREATE DATABASE databasename;
SHOW CREATE TABLE tablename;

CREATE TABLE int_test(id INT); /*创建一个包含整数型id字段的表*/
SELECT * FROM int_test; /*查看int_test表中的数据*/
INSERT INTO int_test VALUES (0),(-1); /*往int_test表中插入数据*/
INSERT INTO int_test VALUES (0),(-1),(1.1),(1234567890),(12345678901),(-12345678901); /*往int_test表中插入数据，由于表中id为整型，因此会报错*/

SHOW WARNINGS; /*显示警告信息*/

CREATE TABLE f_test(a FLOAT(38,30),b DECIMAL(38,30)); /*创建包含浮点型a和定点数型b两个字段的表f_test*/
INSERT INTO f_test VALUES (123.456,123.456),(123450.0000000000000000000000000001,123450.0000000000000000000000000001); /*往表中插入记录（比较FLOAT和DECIMAL数据类型的差别*/
SELECT * FROM f_test \G /*查看f_test表*/

CREATE TABLE bit_test(id BIT(8)); /*创建一个包含位类型数据id的表bit_test*/
INSERT INTO bit_test VALUES (11),(b'11'); /*插入记录，b'11'表示二进制数11*/
SELECT id+0 FROM bit_test; /*查看f_test表中的id数据*/
SELECT BIN(id+0) FROM bit_test; /*以二进制方式查看f_test表中的id数据*/

CREATE TABLE d_test(f_date DATE,f_datetime DATETIME,f_timestamp TIMESTAMP,f_time TIME,f_year YEAR); /*创建包含各种时间日期类型字段的表d_test*/
SELECT CURDATE(),NOW(),NOW(),time(NOW()),YEAR(NOW()) \G /*显示当前时间信息*/
INSERT INTO d_test VALUES (CURDATE(),NOW(),NOW(),time(NOW()),YEAR(NOW())); /*在d-test表中插入当前时间信息*/

CREATE TABLE user(id INT,name VARCHAR(20)); /*创建一个包含整型id和字符串型name字段的表user*/
INSERT INTO user VALUES (1,'bob'),(2,'petter'),(3,'a123456789'); /*在user表中插入记录*/
INSERT INTO user VALUES (1,'bob'),(2,'petter'),(3,'a123456789123456789123456789'); /*在user表中插入记录，由于字符串超过了定义长度，因此会报错*/

SHOW TABLES; /*查看当前数据库中所有的表和视图*/
DESCRIBE table_name; /*查看指定表的信息，包括字段名、字段数据类型等，DESCRIBE可简写成DESC*/
SHOW CREATE TABLE table_name \G /*查看指定表创建时的详细信息*/

TRUNCATE TABLE table_name; /*清空表中所有数据记录*/

DROP TABLE table_name; /*删除指定表*/

ALTER TABLE old_table_name RENAME new_table_name; /*修改指定表的名称*/

ALTER TABLE table_name ADD new_var_name var_type; /*在指定表最后面增加一个字段*/
ALTER TABLE table_name ADD new_var_name var_type FIRST; /*在指定表的第一个位置添加一个字段*/
ALTER TABLE table_name ADD new_var_name var_type AFTER var_name; /*在指定表的指定字段之后添加一个字段*/

ALTER TABLE table_name DROP var_name; /*删除指定表的指定字段*/

ALTER TABLE table_name MODIFY var_name var_new_type; /*修改字段的数据类型*/
ALTER TABLE table_name CHANGE old_var_name new_var_name old_var_type; /*只修改字段的名称*/
ALTER TABLE table_name CHANGE old_var_name new_var_name new_var_type; /*同时修改字段的名字和属性*/

ALTER TABLE table_name MODIFY var_name var_type FIRST; /*将指定字段移动到表的第一个位置*/
ALTER TABLE table_name MODIFY var_name1 var_type1 AFTER var_name2; /*将指定字段移动到另一个指定字段之后*/

ALTER TABLE table_name ADD new_var_name var_type NOT NULL AFTER var_name; /*添加字段并设置非空约束*/
ALTER TABLE table_name ADD new_var_name var_type DEFAULT default_value AFTER var_name; /*添加字段并设置默认值*/
ALTER TABLE table_name ADD new_var_name var_type NOT NULL DEFAULT default_value AFTER var_name; /*添加字段并设置非空约束、默认值*/

ALTER TABLE table_name ADD new_var_name var_type UNIQUE; /*插入新变量并设置唯一值约束*/
ALTER TABLE table_name ADD UNIQUE(var_name); /*为指定字段设置唯一值约束*/
ALTER TABLE table_name ADD CONSTRAINT uk_name UNIQUE(var_name); /*为指定字段添加唯一值约束并给该约束命名为uk_name*/

ALTER TABLE table_name ADD var_name var_type PRIMARY KEY FIRST; /*加入变量并设置为主键，这里将新变量放在最前面了*/
ALTER TABLE table_name ADD PRIMARY KEY(var_name); /*将指定变量设置为主键*/
ALTER TABLE table_name ADD CONSTRAINT pk_name PRIMARY KEY(var_name); /*将指定变量设置为主键并给该主键命名为uk_name*/
ALTER TABLE table_name DROP PRIMARY KEY; /*删除主键*/

ALTER TABLE table_name ADD PRIMARY KEY(var_name1,var_name2,...); /*设置多字段主键*/
ALTER TABLE table_name ADD CONSTRAINT pk_name PRIMARY KEY(var_name1,var_name2,...); /*设置多字段主键并为主键命名*/

ALTER TABLE table_name ADD var_name var_type PRIMARY KEY AUTO_INCREMENT FIRST; /*添加新字段并设置为自动增加（类型必须为整数且同时会被设置为主键）*/

ALTER TABLE table_name ADD CONSTRAINT fk_name FOREIGN KEY(var_name) REFERENCES table_name2(var_name2); /*设置外键并给外键命名（父表为table_name2，父表主键为var_name2）*/
ALTER TABLE table_name DROP FOREIGN KEY fk_name; /*删除外键，外键名称为fk_name*/

--创建表时同时创建普通索引
CREATE TABLE table_name (var1_name var1_type,...,varn_name varn_type,
						 INDEX|KEY [index_name] (var_name [(index_length)] [ASC|DESC])); /*index_name为索引名，index_length为索引长度*/

EXPLAIN SELECT * FROM table_name WHERE var_name=var_value \G /*可用于检验table_name表中var_name字段的索引对象是否被启用，var_value是对应var的取值*/

DROP INDEX index_name ON table_name; /*删除索引*/

--在已经存在的表上创建普通索引
CREATE INDEX index_name ON table_name (var_name [(index_length)] [ASC|DESC]);
ALTER TABLE table_name ADD INDEX|KEY index_name (var1_name [(index_length)] [ASC|DESC]);

--创建表时同时创建唯一索引
CREATE TABLE table_name (var1_name var1_type,...,varn_name varn_type,
						 UNIQUE INDEX|KEY [index_name] (var_name [(index_length)] [ASC|DESC]));
						 
--在已经存在的表上创建唯一索引
CREATE UNIQUE INDEX index_name ON table_name (var_name [(index_length)] [ASC|DESC]);
ALTER TABLE table_name ADD UNIQUE INDEX|KEY index_name (var_name [(index_length)] [ASC|DESC]);

--创建表时同时创建全文索引
CREATE TABLE table_name (var1_name var1_type,...,varn_name varn_type,
						 FULLTEXT INDEX|KEY [index_name] (var_name [(index_length)] [ASC|DESC])) ENGINE=MyISAM; /*ENGINE=MyISAM指定存储引擎*/
						 
--在已经存在的表上创建全文索引
CREATE FULLTEXT INDEX index_name ON table_name (var1_name [(index_length)] [ASC|DESC]);
ALTER TABLE table_name ADD FULLTEXT INDEX|KEY index_name (var_name [(index_length)] [ASC|DESC]);

--创建表时同时创建多列索引
CREATE TABLE table_name (var1_name var1_type,...,varn_name varn_type,
						 INDEX|KEY [index_name] (var_name_1 [(index_length_1)] [ASC|DESC],
						 ...
						 var_name_k [(index_length_k)] [ASC|DESC]));
						 
-- 在已经存在的表上创建多列索引
CREATE INDEX index_name ON table_name (var_name_1 [(index_length_1)] [ASC|DESC],...,var_name_k [(index_length_k)] [ASC|DESC]);
ALTER TABLE table_name ADD INDEX|KEY index_name (var_name_1 [(index_length_1)] [ASC|DESC],...,var_name_k [(index_length_k)] [ASC|DESC]);

CREATE VIEW view_name AS query_command; /*创建视图，query_command为查询语句*/

SELECT * FROM view_name; /*利用建立好的视图进行查询操作*/

DROP VIEW view_name1,view_name2,....,view_namek; /*删除视图*/

CREATE VIEW view_name AS SELECT constant_value; /*创建常量视图，constant_value为常量值*/

CREATE VIEW view_name AS SELECT function_name(var_name) FROM table_name; /*创建使用函数的视图，function_name为函数名称（如SUM,MIN,MAX,COUNT等）*/

CREATE VIEW view_name AS SELECT var_name FROM table_name ORDER BY var_name_for_order [ASC|DESC]; /*创建视图实现排序功能*/

/*创建视图实现表内连接查询例子，找出组别为2的人的姓名（students表中存有姓名和group_id，groups表中存有group_id和group_name）*/
CREATE VIEW view_test AS SELECT s.name FROM students AS s,groups AS g WHERE s.group_id=g.group_id AND g.id=2; 


/*创建视图实现表外连接（LEFT JOIN和RIGHT JOIN）查询例子，找出组别为2的人的姓名（students表中存有姓名和group_id，groups表中存有group_id和group_name）*/
CREATE VIEW view_test AS SELECT s.name FROM students AS s LEFT JOIN groups AS g ON s.group_id=g.id WHERE g.id=2; 

/*创建视图实现子查询例子，找出其组别在groups表中出现过的学生的名字*/
CREATE VIEW var_test AS SELECT s.name FROM students AS s WHERE s.group_id IN (SELECT id FROM groups);

/*创建视图实现记录联合查询（UNION和UNION ALL）*/
CREATE VIEW view_name AS SELECT var_name1,...,var_namek FROM table_name1 UNION [ALL] SELECT var_name1,...,var_namek FROM table_name2;

SHOW TABLE STATUS [FROM database_name] [LIKE 'table|view_name'] \G /*查看表和视图的详细信息*/

SHOW CREATE VIEW view_name; /*查看视图创建时的信息*/

DESCRIBE|DESC view_name; /*查看视图的设计信息*/

--利用MySQL自动创建的information_schema数据库中的views表查看视图信息
USE information_schema;
SELECT * FROM views WHERE TABLE_NAME='view_name' \G

CREATE OR REPLACE VIEW view_name AS query_command; /*替换现有视图，相当于先删除原有视图，再创建一个同名的视图*/

ALTER VIEW view_name AS query_command; /*修改视图*/

--利用视图在基本表中插入数据
INSERT INTO view_name [var1_name,var2_name,...,vark_name] VALEUS (var1_value,var2_value,...,vark_value),...(var1_value,var2_value,...,vark_value);

DELETE FROM view_name WHERE var_name=var_value; /*通过视图删除基本表中的记录*/

UPDATE view_name SET var1_name=var1_value WHERE var2_name=var2_value; /*通过视图更新基本表中的数据*/

--创建有一条执行语句的触发器，trigger_name为触发器名字，trigger_event为触发事件（DELETE、INSERT、UPDATE）,trigger_stmt为激活触发器后要执行的语句
CREATE TRIGGER trigger_name BEFORE|AFTER trigger_event ON table_name FOR EACH ROW trigger_stmt;

--创建包含多条执行语句的触发器,trigger_stmt1;...;trigger_stmtk为触发器激活之后要执行的多条语句，以分号隔开
--由于MySQL中默认以分号作为命令结束符号，因此用DELIMITER $$将结束符号设置成“$$”，结束之后再用DELIMITER还原默认结束符号，更改或还原结束符号时DELIMITER与符号/分号直接的空格不能省略！
DELIMITER $$ CREATE TRIGGER trigger_name BEFORE|AFTER trigger_event ON table_name FOR EACH ROW BEGIN trigger_stmt1;...;trigger_stmtk; END $$ DELIMITER ;

SHOW TRIGGERS \G /*查看当前存在的触发器*/

--利用MySQL自动创建的information_schema数据库中的trigger表查看触发器信息
USE information_schema;
SELECT * FROM TRIGGERS \G
SELECT * FROM TRIGGERS WHERE TRIGGER_NAME='trigger_name' \G

DROP TRIGGER trigger_event; /*删除触发器*/
 
--在表中插入记录
INSERT INTO table_name (var1_name,var2_name,...,vark_name) VALUES (var1_value,var2_value,...,vark_value),...,(var1_value,var2_value,...,vark_value); /*指定需要插入的字段*/
INSERT INTO table_name VALUES (var1_value,var2_value,...,varn_value),...,(var1_value,var2_value,...,varn_value); /*所有字段都插入值，不需要指定字段*/

--在一个表中插入从另一个表的查询结果
INSERT INTO table1_name (var11_name,...,var1k_name) SELECT var21_name,...,var2kname FROM table2_name [WHERE condition];

--更新特定数据记录
UPDATE table_name SET var1=var1_value,var2=var2_value,...,vark=vark_value WHERE condition; /*condition用于指定更新条件*/
UPDATE t_dept3 SET loc='AlocNew' WHERE dname='A'; /*例子，将t_dept3表中dname字段值为A的记录的loc字段值改为AlocNew*/

--更新所有数据记录
UPDATE table_name SET var1=var1_value,var2=var2_value,...,vark=vark_value [WHERE condit];
UPDATE t_dept3 SET loc='This Is Location' WHERE deptno<10; /*例子*/
UPDATE t_dept3 SET loc='This Is Another Location'; /*例子*/

DELETE FROM table_name [WHERE condition]; /*删除记录*/

SELECT var1_name,...,vark_name FROM table_name [WHERE condition]; /*查询记录*/

SELECT DISTINCT var_name FROM table_name; /*查询不重复记录*/

--简单数学运算（+、-、*、/、%（求余））结果查询例子
SELECT sal*12 FROM t_employee; /*在t_employee表中查询sal字段的12倍*/

--为查询的字段以新的字段名称显示
SELECT var1_name [AS] var1_name_new,...,vark_name [AS] vark_name_new FROM table_name; /*AS是可选的，不写也行*/
--从t_employee表中查询ename字段和sal字段的12倍并以yearsalary名称显示
SELECT ename,sal*12 AS yearssalary FROM t_employee;
SELECT ename,sal*12 yearssalary FROM t_employee;

--设置显示格式数据查询示例
SELECT CONCAT(ename,'雇员的年薪为：',sal*12) [AS] yearssalary FROM t_employee;

--MySQL中的比较运算符包括：>、<、=、!=（<>）、>=、<=
--MySQL中的逻辑运算符包括：AND（&&）、OR（||）、XOR、NOT（!）

SELECT var1_name,...,vark_name FROM table_name WHERE var_name [NOT] BETWEEN value1 AND value2; /*设置查询条件为var_name字段的值（不）在value1和value2之间*/

SELECT var1_name,...,vark_name FROM table_name WHERE var_name IS [NOT] NULL; /*设置查询条件为var_name（不）是空值*/
SELECT var1_name,...,vark_name FROM table_name WHERE [NOT] var_name IS NULL; /*设置查询条件为var_name（不）是空值*/

SELECT var1_name,...,vark_name FROM table_name WHERE var_name [NOT] IN (value1,...,valuen); /*设置查询条件为var_name字段的值（不）在某个集合中*/
SELECT var1_name,...,vark_name FROM table_name WHERE [NOT] var_name IN (value1,...,valuen); /*设置查询条件为var_name字段的值（不）在某个集合中*/
--注意：条件集合中有NULL，查询语句中没有NOT关键字时对查询结果没有影响，若查询语句中有NOT关键字，则查询结果为空

--MySQL中的通配符：_为单个字符通配符，%为任意多个字符通配符，%%代表所有记录都匹配
SELECT var1_name,...,vark_name FROM table_name WHERE [NOT] var_name LIKE vlaue; /*设置查询条件为跟value（不）相匹配的值*/
SELECT var1_name,...,vark_name FROM table_name WHERE var_name [NOT] LIKE vlaue; /*设置查询条件为跟value（不）相匹配的值*/
SELECT * FROM t_employee WHERE job LIKE 'A%'; /*查询t_employee表中job字段中以A开头的记录*/
SELECT * FROM t_employee WHERE job LIKE '_L%'; /*查询t_employee表中job字段中第二个字母为L的记录*/

--排序查询结果，排序时NULL按照最小值处理
SELECT var1_name,...,vark_name FROM table_name WHERE condition ORDER BY var1_name [ASC|DESC],...,vark_name [ASC|DESC];

--/*限制查询行数，offset_start为起始位置，row_count为需要显示的行数，offset_start默认为0，即从第一条开始*/
SELECT var1_name,...,vark_name FROM table_name WHERE condition LIMIT offset_start,row_count;
SELECT * FROM t_employee WHERE comm IS NULL LIMIT 2; /*查询comm字段为空值的记录，仅显示前两行*/
SELECT * FROM t_employee WHERE comm IS NULL LIMIT 0,2; /*查询comm字段为空值的记录，仅显示前两行*/

SELECT * FROM t_employee WHERE comm IS NULL ORDER BY hiredate LIMIT 5,5; /*查询t_employee表中comm为空值的记录，按hiredate升序排列，从第6条开始，查询5条记录*/

SELECT COUNT(*) FROM table_name WHERE condition; /*此时计数不会忽略NULL值*/
SELECT COUNT(var_name) FROM table_name WHERE condition; /*此时计数会忽略var_name中的NULL值*/

SELECT AVG(var_name) FROM table_name WHERE condition; /*计算平均值时会忽略var_name中的NULL值*/

SELECT SUM(var_name) FROM table_name WHERE condition; /*求和时会忽略var_name中的NULL值*/

SELECT MAX|MIN(var_name) FROM table_name WHERE condition; /*求最大最小值时会忽略var_name中的NULL值*/

--单分组数据查询
SELECT function_name() FROM table_name WHERE condition GROUP BY var_name; /*简单分组查询*/
SELECT GROUP_CONCAT(var_name) FROM table_name WHERE condition GROUP BY var1_name; /*按var1_name分组显示var_name的取值*/
SELECT deptno,GROUP_CONCAT(ename) enames,COUNT(ename) num_ename FROM t_employee GROUP BY deptno; /*查询每个deptno下面有哪些ename并计数*/

--多分组查询
SELECT GROUP_CONCAT(var_name),function_name() FROM table_name WHERE condition GROUP BY var1_name,var2_name,...,vark_name;
/*例：按deptno和hiredate分组查询ename的值并计数*/
SELECT deptno,hiredate,GROUP_CONCAT(ename) enames,COUNT(ename) num_ename FROM t_employee GROUP BY deptno,hiredate;

--分组条件查询
SELECT function_name() FROM table_name WHERE condition GROUP BY var1_name,var2_name,...,vark_name HAVING condition;
SELECT deptno,AVG(sal) avg_sal FROM t_employee GROUP BY deptno; /*按deptno分组查询平均sal*/
/*例：按deptno分组计算平均sal，然后显示平均sal大于2000的组别以及这些组别里的ename及其计数*/
SELECT deptno,AVG(sal) avg_sal,GROUP_CONCAT(ename) enames,COUNT(ename) num_ename FROM t_employee GROUP BY deptno HAVING AVG(sal)>2000;

--自连接查询
SELECT var1_name var2_name ... vark_name FROM join_table_name1 INNER JOIN join_table_name2 [INNER JOIN join_table_name] ON join_condition;
SELECT e.ename employeename,e.job FROM t_employee e; /*从t_employee表中查询ename和job*/
/*例子：t_employee表包含了雇员信息和领导信息（既是雇员表也是领导表），现在要找出各个雇员的职位及其领导
其中mgr字段为领导编号，empno字段为雇员自身编号。有如下两种方式*/
SELECT e.ename employeename,e.job,l.ename loadername FROM t_employee e,t_employee l WHERE e.mgr=l.empno;
SELECT e.ename employeename,e.job,l.ename loadername FROM t_employee e INNER JOIN t_employee l ON e.mgr=l.empno;

--等值连接查询
SELECT e.empno,e.ename,e.job FROM t_employee e; /*从t_employee表中查询empno、ename、job三个字段*/
/*例子：t_employee表包含了雇员信息，t_dept表里包含了部门信息，两个表中都有部门编号列deptno，
现需要查询雇员的工作信息及其部门信息。有如下两种方式*/
SELECT e.empno,e.ename,e.job,d.dname,d.loc FROM t_employee e INNER JOIN t_dept d ON e.deptno=d.deptno;
SELECT e.empno,e.ename,e.job,d.dname,d.loc FROM t_employee e,t_dept d WHERE e.deptno=d.deptno;
/*例子：在上个例子的基础上，还需要查询领导的信息（领导信息也在雇员信息里面，所以领导表就是雇员表，
其中mgr字段为领导编号，empno字段为雇员编号*/
SELECT e.empno,e.ename employeename,e.sal,e.job,l.ename loadername FROM t_employee e INNER JOIN t_employee l ON e.mgr=l.empno; /*并未查询部门信息*/
SELECT e.empno,e.ename employeename,e.sal,e.job,l.ename loadername,d.dname,d.loc FROM t_employee e INNER JOIN t_employee l ON e.mgr=l.empno INNER JOIN t_dept d ON l.deptno=d.deptno;
SELECT e.empno,e.ename employeename,e.sal,e.job,l.ename loadername,d.dname,d.loc FROM t_employee e,t_employee l,t_dept d WHERE e.mgr=l.empno AND l.deptno=d.deptno;

--不等连接查询
/*例子：现在需要查询雇员编号大于领导编号的雇员的信息及其领导信息（雇员表和领导表是同一个表），mgr字段为领导编号，empno字段为雇员自身编号。有如下两种方式）*/
SELECT e.ename employeename,e.job,l.ename loadername FROM t_employee e INNER JOIN t_employee l ON e.mgr=l.empno AND e.empno>l.empno;
SELECT e.ename employeename,e.job,l.ename loadername FROM t_employee e,t_employee l WHERE e.mgr=l.empno AND e.empno>l.empno;

--mysql中用命令行复制表结构的方法主要有一下几种: 
--1.只复制表结构到新表
CREATE TABLE 新表名 SELECT * FROM 旧表名 WHERE 1=2;
--或
CREATE TABLE 新表名 LIKE 旧表名;
--注意上面两种方式，前一种方式复制时主键类型和自增方式是不会复制过去的，而后一种方式是把旧表的所有字段类型都复制到新表。
--2.复制表结构及数据到新表
CREATE TABLE 新表名 SELECT * FROM 旧表名;
--3.复制旧表的数据到新表(假设两个表结构一样) 
INSERT INTO 新表名 SELECT * FROM 旧表名;
--4.复制旧表的数据到新表(假设两个表结构不一样)
INSERT INTO 新表名(字段1,字段2,.......) SELECT 字段1,字段2,...... FROM 旧表名;

--外连接查询
SELECT var1_name,...,vark_name FROM join_table_name1 LEFT|RIGHT|FULL [OUTER] JOIN join_table_name2 ON join_condition;
/*左外连接例子：查询雇员信息和领导信息（雇员表和领导表是同一个表），mgr字段为领导编号，empno字段为雇员自身编号。
其中KING雇员没有领导，也要显示出来（如果用内连接则不能实现）*/
SELECT e.ename employeename,e.job,l.ename loadername FROM t_employee e LEFT JOIN t_employee l ON e.mgr=l.empno;
/*右外连接例子：查询雇员信息和其部门信息，雇员表中cjgong雇员的部门信息在部门表中没有，但仍要将其显示出来（如果用内连接则不能实现）*/
SELECT e.ename employeename,e.job,d.dname,d.loc FROM t_employee e RIGHT JOIN t_dept d ON e.deptno=d.deptno;

--合并查询数据记录（UNION会丢弃重复值，UNION不会丢弃重复值）
SELECT var1_name,...,vark_name FROM table1_name UNION|UNION ALL SELECT var1_name,...,vark_name FROM table2_name UNION|UNION ALL SELECT var1_name,...,vark_name FROM table3_name ...
SELECT * FROM t_cstudent UNION SELECT * FROM t_mstudent; /*将两个表的查询结果去重复值之后合并*/
SELECT * FROM t_cstudent UNION ALL SELECT * FROM t_mstudent; /*将两个表的查询结果直接全部合并*/

--子查询
SELECT sal FROM t_employee WHERE ename='SMITH'; /*从t_employee表中查询SMITH的sal*/
/*例子：通过子查询显示sal比SMITH大的所有人的记录*/
SELECT * FROM t_employee WHERE sal>(SELECT sal FROM t_employee WHERE ename='SMITH');
/*例子：通过子查询显示job和sal都与SMITH一样的所有人的记录*/
SELECT * FROM t_employee WHERE (sal,job)=(SELECT sal,job FROM t_employee WHERE ename='SMITH');
/*例子：通过子查询显示雇员表中的雇员信息，要求显示的是雇员表中的部门在部门表中有出现的雇员*/
SELECT * FROM t_employee WHERE deptno IN (SELECT deptno FROM t_dept);
/*例子：通过子查询显示雇员表中的雇员信息，要求显示的是雇员表中的部门在部门表中没有出现的雇员*/
SELECT * FROM t_employee WHERE deptno NOT IN (SELECT deptno FROM t_dept);
/*例子：查询雇员表的信息，要求这些雇员的sald高于（不低于或低于）job为MANAGER的sal（只要大于所有MANAGER的sal中最小的一个都满足条件）*/
SELECT * FROM t_employee WHERE sal > ANY(SELECT sal FROM t_employee WHERE job='MANAGER');
SELECT * FROM t_employee WHERE sal >= ANY(SELECT sal FROM t_employee WHERE job='MANAGER');
SELECT * FROM t_employee WHERE sal < ANY(SELECT sal FROM t_employee WHERE job='MANAGER');
/*例子：查询雇员表的信息，要求这些雇员的sald大于（大于等于或小于）所有job为MANAGER的sal（需要大于所有MANAGER的sal中最大值）*/
SELECT * FROM t_employee WHERE sal > ALL(SELECT sal FROM t_employee WHERE job='MANAGER');
SELECT * FROM t_employee WHERE sal >= ALL(SELECT sal FROM t_employee WHERE job='MANAGER');
SELECT * FROM t_employee WHERE sal <= ALL(SELECT sal FROM t_employee WHERE job='MANAGER');

SELECT * FROM t_employee e,t_dept d WHERE e.deptno=d.deptno; /*根据雇员表和部门表查询记录，要求两表中的部门编号相互匹配*/
/*例子：从部门表t_dept中查询部门信息，要求该部门在雇员表t_employee中没有雇员记录（即雇员表中没有出现过该部门信息）*/
SELECT * FROM t_dept d WHERE NOT EXISTS(SELECT * FROM t_employee WHERE deptno=d.deptno);
/*例子：从部门表t_dept中查询部门信息，要求该部门在雇员表t_employee中有雇员记录（即雇员表中有出现过该部门信息）*/
SELECT * FROM t_dept d WHERE EXISTS(SELECT * FROM t_employee WHERE deptno=d.deptno);

/*例子：查询雇员表t_employee中各部门的deptno、dname、loc、雇员人数、平均sal。两种方法：*/
--第一种方法，通过内连接查询
SELECT d.deptno,d.dname,d.loc,COUNT(e.empno) number,AVG(e.sal) average FROM t_employee e INNER JOIN t_dept d ON e.deptno=d.deptno GROUP BY d.deptno;
--第二种方法，通过子查询
SELECT d.deptno,d.dname,d.loc,e.number,e.average FROM t_dept d INNER JOIN (SELECT deptno dno,COUNT(empno) number,AVG(sal) average FROM t_employee GROUP BY deptno DESC) e ON d.deptno=e.dno;

# '''
























