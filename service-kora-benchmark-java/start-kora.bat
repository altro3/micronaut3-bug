@echo off
title Kora Service (Cores 0-7)

start "Kora" /affinity FF java ^
  -Xms16m ^
  -Xmx192m ^
  -Xss256k ^
  -XX:MaxDirectMemorySize=48m ^
  -XX:MaxMetaspaceSize=96m ^
  -XX:ActiveProcessorCount=8 ^
  -XX:+UseG1GC ^
  -XX:+G1PeriodicGCInvokesConcurrent ^
  -XX:G1PeriodicGCInterval=3000 ^
  -XX:MinHeapFreeRatio=20 ^
  -XX:MaxHeapFreeRatio=40 ^
  -XX:MaxGCPauseMillis=20 ^
  -XX:+UseStringDeduplication ^
  -XX:+UnlockExperimentalVMOptions ^
  -XX:+UseCompactObjectHeaders ^
  -XX:+EliminateAllocations ^
  -XX:-ShrinkHeapInSteps ^
  -XX:G1ReservePercent=5 ^
  -Djdk.trackAllThreads=false ^
  -Dfile.encoding=UTF-8 ^
  -Dconfig.override_with_env_vars=true ^
  -Dcom.sun.management.jmxremote=false ^
  -Dcom.sun.management.jmxremote.ssl=false ^
  -Dcom.sun.management.jmxremote.port=0 ^
  -Dcom.sun.management.jmxremote.authenticate=false ^
  -XX:+DisableAttachMechanism ^
  -Djava.rmi.server.hostname=127.0.0.1 ^
  -XX:SharedArchiveFile=service.jsa ^
  --add-opens=java.base/java.lang=ALL-UNNAMED ^
  --add-modules jdk.incubator.vector ^
  -jar service-kora.jar
