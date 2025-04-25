FROM alpine:20250108

RUN apk update && apk upgrade --no-cache
RUN apk update && apk add openjdk17-jre ffmpeg

COPY build/libs/micronaut3-bug.jar app.jar

ENTRYPOINT ["java", "-Dfile.encoding=UTF-8", "-jar","/app.jar"]
