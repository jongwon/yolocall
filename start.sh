#!/bin/bash
exec 1>logs/application.log

nohup uvicorn app1:app --host=0.0.0.0 --port=8001 --reload >> output.log &



