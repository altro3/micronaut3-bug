package com.micronaut.bug.service;

import org.bytedeco.javacv.FFmpegFrameFilter;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;

@Service
public class MyService {

    public void uploadVideo(MultipartFile file) throws Exception {
        var croppedVideoFile = getCroppedVideoFile(file);
    }

    private File getCroppedVideoFile(MultipartFile videoFile) throws Exception {
        var frameGrabber = new FFmpegFrameGrabber(videoFile.getInputStream());
        frameGrabber.start();
        var frameFilter = new FFmpegFrameFilter("crop=100:200:0:0", frameGrabber.getImageWidth(), frameGrabber.getImageHeight());
        frameFilter.start();

        // Временный файл, в который кладётся обработанное видео. После загрузки в MyTarget удаляется
        var croppedVideoFile = new File("tmp/${UUID.randomUUID()}.mp4");
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

        try {
            // Обработка видео и запись во временный файл
            recordVideo(frameGrabber, frameFilter, frameRecorder);
        } finally {
            frameRecorder.stop();
            frameFilter.stop();
            frameGrabber.stop();
        }

        return croppedVideoFile;
    }

    private void recordVideo(FFmpegFrameGrabber frameGrabber, FFmpegFrameFilter frameFilter, FFmpegFrameRecorder frameRecorder) throws Exception {
        Frame frame;
        do {
            frame = frameGrabber.grab();
            frameFilter.push(frame);
            frameRecorder.record(frameFilter.pull());
        } while (frame != null);
    }
}
