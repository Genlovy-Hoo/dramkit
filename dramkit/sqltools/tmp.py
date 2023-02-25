# -*- coding: utf-8 -*-

import os
from impala.dbapi import connect
from krbcontext import krbcontext
keytab_path = os.path.split(os.path.realpath(__file__))[0] + '/tmp_hive.keytab'
principal = 'hive/master@HIVE.COM'
with krbcontext(using_keytab=True,
                principal=principal,
                keytab_file=keytab_path):
    conn = connect(host='192.168.118.128',
                   port=10000,
                   auth_mechanism='GSSAPI',
                   kerberos_service_name='hive')
    cursor = conn.cursor()
    cursor.execute('show databases')
    for row in cursor:
        print(row)