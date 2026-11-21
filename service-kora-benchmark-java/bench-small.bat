@echo off
title Bombardier SMALL (Cores 8-15)
set GOMAXPROCS=8

powershell -NoProfile -Command "$p = Start-Process .\bombardier.exe -ArgumentList '-c 128 -d 60s --latencies -m POST -H \"Content-Type: application/json\" -H \"Accept: application/json\" -f small.json http://127.0.0.1:8090/bench/small' -NoNewWindow -PassThru; $p.ProcessorAffinity = 0xFF00; $p.WaitForExit()"

pause
