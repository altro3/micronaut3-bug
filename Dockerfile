FROM ubuntu:24.10
#FROM debian:bookworm-slim

RUN apt update && apt upgrade -y

RUN apt update && apt install -y --no-install-recommends openjdk-17-jre

RUN apt update && apt install -y ffmpeg

COPY build/libs/micronaut3-bug.jar app.jar

ENTRYPOINT ["java", "-Dfile.encoding=UTF-8", "-jar","/app.jar"]
