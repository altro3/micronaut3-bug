package com.micronaut.bug.service;

import com.github.kokorin.jaffree.StreamType;
import com.github.kokorin.jaffree.ffmpeg.ChannelInput;
import com.github.kokorin.jaffree.ffmpeg.ChannelOutput;
import com.github.kokorin.jaffree.ffmpeg.FFmpeg;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;

@Service
public class MyService {

    public void uploadVideo(MultipartFile file) throws Exception {
        var croppedVideoFile = getCroppedVideoFile(file);
    }

    private File getCroppedVideoFile(MultipartFile videoFile) throws Exception {


        var tmpDir = System.getProperty("java.io.tmpdir");
        var rqFile = Path.of(tmpDir, UUID.randomUUID().toString() + System.nanoTime() + ".mp4");
        Files.copy(videoFile.getInputStream(), rqFile, REPLACE_EXISTING);
        // Временный файл, в который кладётся обработанное видео. После загрузки в MyTarget удаляется
        var croppedVideoFile = Path.of(tmpDir, UUID.randomUUID().toString() + System.nanoTime() + ".mp4").toFile();
//        var croppedVideoFile = File.createTempFile(UUID.randomUUID().toString(), ".mp4");
//        croppedVideoFile.deleteOnExit();


        try (var out = new FileOutputStream(croppedVideoFile)) {
            var result = FFmpeg.atPath()
                .addInput(ChannelInput.fromChannel(new FileInputStream(rqFile.toFile()).getChannel()))
                .setFilter(StreamType.VIDEO, "crop=200:400:0:0")
                .addOutput(
                    ChannelOutput.toChannel(UUID.randomUUID().toString() + System.nanoTime() + ".mp4", out.getChannel())
                        .setFrameSize(200, 400)
                )
                .execute();
        }
        return croppedVideoFile;

/*
        var frameGrabber = new FFmpegFrameGrabber(videoFile.getInputStream());
        frameGrabber.start();
        var frameFilter = new FFmpegFrameFilter("crop=100:200:0:0", frameGrabber.getImageWidth(), frameGrabber.getImageHeight());
        frameFilter.start();

        var frameRecorder = new FFmpegFrameRecorder(croppedVideoFile, 100, 200, frameGrabber.getAudioChannels());
        frameRecorder.setAudioBitrate(frameGrabber.getAudioBitrate());
        frameRecorder.setAudioCodec(frameGrabber.getAudioCodec());
        frameRecorder.setAudioCodecName(frameGrabber.getAudioCodecName());
        frameRecorder.setAudioMetadata(frameGrabber.getAudioMetadata());
        frameRecorder.setAudioOptions(frameGrabber.getAudioOptions());
        frameRecorder.setVideoBitrate(frameGrabber.getVideoBitrate());
        frameRecorder.setVideoCodec(frameGrabber.getVideoCodec());
        frameRecorder.setVideoCodecName(frameGrabber.getVideoCodecName());
        frameRecorder.setVideoMetadata(frameGrabber.getVideoMetadata());
        frameRecorder.setVideoOptions(frameGrabber.getVideoOptions());
        frameRecorder.setMaxDelay(frameGrabber.getMaxDelay());
        frameRecorder.setMetadata(frameGrabber.getMetadata());
        frameRecorder.setOptions(frameGrabber.getOptions());
        frameRecorder.setFrameNumber(frameGrabber.getFrameNumber());
        frameRecorder.setFrameRate(frameGrabber.getFrameRate());
        frameRecorder.setSampleRate(frameGrabber.getSampleRate());
        frameRecorder.setFormat(frameGrabber.getFormat());
        frameRecorder.start();
*/

/*
        try {
            // Обработка видео и запись во временный файл
            recordVideo(frameGrabber, frameFilter, frameRecorder);
        } finally {
            frameRecorder.stop();
            frameFilter.stop();
            frameGrabber.stop();
        }

        return croppedVideoFile;
*/
    }

/*
    private void recordVideo(FFmpegFrameGrabber frameGrabber, FFmpegFrameFilter frameFilter, FFmpegFrameRecorder frameRecorder) throws Exception {
        Frame frame;
        do {
            frame = frameGrabber.grab();
            frameFilter.push(frame);
            frameRecorder.record(frameFilter.pull());
        } while (frame != null);
    }
*/
}
