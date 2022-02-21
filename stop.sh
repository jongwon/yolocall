#!/bin/bash

app_name=uvicorn

ps x | grep -v grep | grep $app_name | awk '{ print $1 }' > pidlist

while read pid;
do
        echo "process found ==> ${pid}"
        `kill ${pid}`
done < pidlist


ps x | grep -v grep | grep 'multiprocessing.spawn' | awk '{ print $1 }' > pidlist2
while read pid2;
do
        echo "multiprocessing.spawn process found ==> ${pid2}"
        `kill ${pid2}`
done < pidlist2


echo "kill all process"
