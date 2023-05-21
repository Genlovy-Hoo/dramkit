#!/bin/sh
echo "To killing openai_chat_server_new!"
pids=`ps -ef | grep openai_chat_server_new | grep -v grep | awk '{print $2}'`
echo $pids
for pid in $pids
do
kill -9 $pid
echo "kill $pid"
done
