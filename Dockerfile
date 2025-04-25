FROM frolvlad/alpine-glibc:alpine-3.21_glibc-2.41

RUN apk update && apk upgrade --no-cache
RUN apk update && apk add openjdk17-jre ffmpeg

COPY build/libs/micronaut3-bug.jar app.jar

ENTRYPOINT ["java", "-Dfile.encoding=UTF-8", "-jar","/app.jar"]
