#!/bin/sh
echo "To killing openai_chat_server!"
pids=`ps -ef | grep openai_chat_server | grep -v grep | awk '{print $2}'`
echo $pids
for pid in $pids
do
kill -9 $pid
echo "kill $pid"
done
