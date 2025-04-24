FROM debian:bookworm

RUN apt update && apt upgrade -y

RUN apt update && apt install -y --no-install-recommends openjdk-17-jre

RUN apt update && apt install -y libc-dev \
  build-essential yasm automake autoconf \
  libtool pkg-config libcurl4-openssl-dev \
  intltool libxml2-dev libgtk2.0-dev \
  libnotify-dev libglib2.0-dev libevent-dev

COPY ffmpeg-7.1.1 ffmpeg
RUN cd ffmpeg  \
    && ./configure --prefix=/usr \
    && make -j8 \
    && make install
