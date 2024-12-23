'use client'

import React, { useState, useEffect } from 'react';
import FileUpload from '../components/FileUpload';
import ProcessingPage from './ProcessingPage';
import AugmentedVideo from './AugmentedVideo';


function simulateLongProcess(file: File): Promise<boolean> {
  return new Promise((resolve) => {
    console.log(`Processing file: ${file.name}, Size: ${file.size} bytes`);
    const processingTime = 0;

    setTimeout(() => {
      console.log(`Finished processing: ${file.name}`);
      resolve(true);
    }, processingTime);
  });
}

const MainPage: React.FC = () => {
  const [result, setResult] = useState<boolean | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [originalVideoSrc, setOriginalVideoSrc] = useState<string | null>(null);


  const handleFileUpload = async (file: File) => {
    setIsProcessing(true);
    setResult(null);
    setOriginalVideoSrc(URL.createObjectURL(file));

    try {
      const processedResult = await simulateLongProcess(file);
      setResult(processedResult);
    } catch (error) {
      console.error("Error processing file:", error);
    } finally {
      setIsProcessing(false);
    }
  };



  useEffect(() => {
    if (originalVideoSrc && result !== null) {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    }
  }, [originalVideoSrc, result]);

  const roiVideos = {
    leftEye: '/data/002/002_left_eye_roi.mp4',
    mouth: '/data/002/002_mouth_roi.mp4',
    nose: '/data/002/002_nose_roi.mp4',
  };

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [score, setScore] = useState<string | null>(null);


  useEffect(() => {
    const postVideo = async () => {
      try {
        const videoFile = await fetch('/data/002/002_nose_roi.mp4').then(res => res.blob());
        const formData = new FormData();
        formData.append('file', videoFile, '002_nose_roi.mp4');
  
        const response = await fetch('http://localhost:8000/api/models/test', {
          method: 'POST',
          body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to post video');
      }


      const filePath = response.headers.get('File-Path');
      console.log('File path:', filePath);

      const videoBlob = await response.blob();
      const videoUrl = URL.createObjectURL(videoBlob);
      setVideoUrl(videoUrl);
    } catch (error) {
      console.error('Error posting video:', error);
    }
  };

  postVideo();
  }, []);


  return (
    <main className="flex flex-col items-center justify-center p-6">
      <h1 className="text-3xl font-bold mt-24 mb-4">Deepfake Detector</h1>
      <FileUpload onFileUpload={handleFileUpload} setVideoUrl={setVideoUrl} setScore={setScore}/>
      <p className="mt-2 mb-64 text-sm text-gray-600">Upload an image or video to check for deepfakes.</p>
      {originalVideoSrc && result !== null ? (
        <>
          <ProcessingPage originalVideoSrc={originalVideoSrc} roiVideos={roiVideos} />
          {videoUrl && <AugmentedVideo videoUrl={videoUrl} score={score} />}
        </>
      ) : null}
    </main>
  );
};

export default MainPage;