#!/bin/bash

app_name=uvicorn

ps x | grep -v grep | grep $app_name | awk '{ print $1 }' > pidlist

while read pid;
do
        echo "process found ==> ${pid}"
        `kill -9 ${pid}`
done < pidlist

echo "kill all process"

