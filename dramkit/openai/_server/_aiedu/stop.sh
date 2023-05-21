#!/bin/sh
echo "To killing aiedu_chat_web!"
pids=`ps -ef | grep aiedu_chat_web | grep -v grep | awk '{print $2}'`
echo $pids
for pid in $pids
do
kill -9 $pid
echo "kill $pid"
done
