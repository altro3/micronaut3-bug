package com.micronaut.bug.service;

import com.github.kokorin.jaffree.StreamType;
import com.github.kokorin.jaffree.ffmpeg.ChannelInput;
import com.github.kokorin.jaffree.ffmpeg.ChannelOutput;
import com.github.kokorin.jaffree.ffmpeg.FFmpeg;
import com.github.kokorin.jaffree.ffmpeg.Frame;
import com.github.kokorin.jaffree.ffmpeg.FrameConsumer;
import com.github.kokorin.jaffree.ffmpeg.FrameOutput;
import com.github.kokorin.jaffree.ffmpeg.Stream;
import com.github.kokorin.jaffree.ffprobe.FFprobe;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static java.nio.file.StandardOpenOption.CREATE;
import static java.nio.file.StandardOpenOption.READ;
import static java.nio.file.StandardOpenOption.TRUNCATE_EXISTING;
import static java.nio.file.StandardOpenOption.WRITE;

@Service
public class MyService {

    public void uploadVideo(MultipartFile file) throws Exception {
        var croppedVideoFile = getCroppedVideoFile(file);
    }

    private File getCroppedVideoFile(MultipartFile videoFile) throws Exception {


        var tmpDir = System.getProperty("java.io.tmpdir");
        var rqFile = Path.of(tmpDir, UUID.randomUUID().toString() + System.nanoTime() + ".mp4");
        Files.copy(videoFile.getInputStream(), rqFile, REPLACE_EXISTING);
        var croppedVideoFile = Path.of(tmpDir, UUID.randomUUID().toString() + System.nanoTime() + ".mp4");

        try (
            var in = Files.newByteChannel(rqFile, READ);
            var out = Files.newByteChannel(croppedVideoFile, CREATE, WRITE, READ, TRUNCATE_EXISTING)) {
            var result = FFmpeg.atPath()
                .addInput(ChannelInput.fromChannel(in))
                .setFilter(StreamType.VIDEO, "crop=700:600:0:0")
                .addOutput(
                    ChannelOutput.toChannel(UUID.randomUUID().toString() + System.nanoTime() + ".mp4", out)
                        .setFrameSize(700, 600)
                )
                .execute();
        }

        getMultipartIconFile(croppedVideoFile);

        return croppedVideoFile.toFile();
    }

    private void getMultipartIconFile(Path croppedVideoFile) {

        var result = FFprobe.atPath()
            .setShowStreams(true)
            .setInput(croppedVideoFile)
            .execute();

        long durationMillis = 0;
        for (var stream : result.getStreams()) {
            if (stream.getCodecType() == StreamType.VIDEO) {
                durationMillis = (long) (stream.getDuration() * 1000);
                break;
            }
        }

        if (durationMillis == 0) {
            throw new RuntimeException("No video found");
        }

        Integer rotationDegrees = result.getData().getInteger("rotation");
        final BufferedImage[] frameImageBuffer = {null};

        try (var input = Files.newByteChannel(croppedVideoFile, READ)) {
            FFmpeg.atPath()
                .addInput(
                    ChannelInput.fromChannel(input)
                        .setPosition(durationMillis / 2)
                )
                .setFilter(StreamType.VIDEO, "scale=50:50")
                .addOutput(
                    FrameOutput.withConsumer(new FrameConsumer() {
                            @Override
                            public void consumeStreams(List<Stream> streams) {
                            }

                            @Override
                            public void consume(Frame frame) {
                                if (frame == null || frame.getImage() == null) {
                                    return;
                                }
                                if (frameImageBuffer[0] == null) {
                                    frameImageBuffer[0] = frame.getImage();
                                }
                            }
                        })
                        .setFrameCount(StreamType.VIDEO, 1L)
                )
                .execute();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        var tmpDir = System.getProperty("java.io.tmpdir");
        var imageFile = Path.of(tmpDir, UUID.randomUUID().toString() + System.nanoTime() + ".jpg").toFile();

        if (rotationDegrees != null) {
            var rotationImageBuffer = new BufferedImage(frameImageBuffer[0].getWidth(), frameImageBuffer[0].getHeight(), frameImageBuffer[0].getType());
            var imageGraphics = rotationImageBuffer.createGraphics();
            imageGraphics.rotate(
                Math.toRadians(rotationDegrees),
                frameImageBuffer[0].getWidth() / 2.0,
                frameImageBuffer[0].getHeight() / 2.0
            );
            imageGraphics.drawImage(frameImageBuffer[0], null, 0, 0);
            frameImageBuffer[0] = rotationImageBuffer;
        }

        try (var out = new FileOutputStream(imageFile)) {
            ImageIO.write(frameImageBuffer[0], "jpeg", out);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


// var frameImageBuffer = Java2DFrameConverter().getBufferedImage(scaleFrameFilter.pullImage())
//        var iconFrameImageBuffer = alignIconImageBuffer(frameGrabber, frameImageBuffer[0])

    }

/*
    private fun alignIconImageBuffer(
        frameGrabber:FFmpegFrameGrabber,
        frameImageBuffer:BufferedImage,
        ):

    BufferedImage {
        var rotateDegrees = frameGrabber.videoMetadata["rotate"] ?.toDoubleOrNull()
        return rotateDegrees ?.let {
            var rotationImageBuffer =BufferedImage(frameImageBuffer.width, frameImageBuffer.height, frameImageBuffer.type)
            var imageGraphics = rotationImageBuffer.createGraphics()
            imageGraphics.rotate(
                Math.toRadians(rotateDegrees),
                frameImageBuffer.width / 2.0,
                frameImageBuffer.height / 2.0
            )
            imageGraphics.drawImage(frameImageBuffer, null, 0, 0)
            rotationImageBuffer
        } ?:frameImageBuffer
    }
*/
}
