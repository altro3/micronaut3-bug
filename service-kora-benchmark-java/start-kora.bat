@echo off
title Kora Service (Cores 0-7)

start "Kora" /affinity FF java ^
  -Xms8m ^
  -Xmx192m ^
  -XX:MaxDirectMemorySize=64m ^
  -Xss256k ^
  -XX:ActiveProcessorCount=8 ^
  -XX:+UseG1GC ^
  -XX:+G1PeriodicGCInvokesConcurrent ^
  -XX:G1PeriodicGCInterval=3000 ^
  -XX:MinHeapFreeRatio=10 ^
  -XX:MaxHeapFreeRatio=25 ^
  -XX:MaxGCPauseMillis=20 ^
  -XX:+UseStringDeduplication ^
  -XX:+UnlockExperimentalVMOptions ^
  -XX:+UseCompactObjectHeaders ^
  -XX:+EliminateAllocations ^
  -Dfile.encoding=UTF-8 ^
  -Dconfig.override_with_env_vars=true ^
  -Dcom.sun.management.jmxremote=false ^
  -Dcom.sun.management.jmxremote.ssl=false ^
  -Dcom.sun.management.jmxremote.port=0 ^
  -Dcom.sun.management.jmxremote.authenticate=false ^
  -Djava.rmi.server.hostname=127.0.0.1 ^
  -XX:SharedArchiveFile=service.jsa ^
  --enable-native-access=ALL-UNNAMED ^
  --add-opens=java.base/java.lang=ALL-UNNAMED ^
  --add-modules jdk.incubator.vector ^
  -XX:StartFlightRecording=duration=60s,filename=recording.jfr ^
  -XX:NativeMemoryTracking=detail ^
  -jar service-kora.jar
