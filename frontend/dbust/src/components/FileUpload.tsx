import React, { useState, useCallback } from 'react';
import styles from './FileUpload.module.css';
import { ProgressCircleRing, ProgressCircleRoot } from "@/components/ui/progress-circle";
import { HStack } from "@chakra-ui/react";

interface FileUploadProps {
    onFileUpload: (file: File) => void;
    setVideoUrl: (url: string) => void;
    setScore: (score: string) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, setVideoUrl, setScore }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [preview, setPreview] = useState<string | null>(null);
    const [isPreviewShown, setIsPreviewShown] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);

    const handleDragEnter = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.currentTarget === e.target) {
            setIsDragging(false);
        }
    }, []);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    }, []);

    const handleFile = useCallback((file: File) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            setPreview(reader.result as string);
            setIsPreviewShown(true);
            setSelectedFile(file);
        };
        reader.readAsDataURL(file);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    }, [handleFile]);

    const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    }, [handleFile]);

    const handleUpload = useCallback(async () => {
        if (selectedFile) {
            setIsProcessing(true);
            // const startTime = Date.now();
            // console.log(`Upload started at: ${new Date(startTime).toISOString()}`);

            try {
                // File Upload
                // const formData = new FormData();
                // formData.append('file', selectedFile);

                // const response = await fetch('http://localhost:8000/api/files/upload', {
                //     method: 'POST',
                //     body: formData,
                // });

                // const endTime = Date.now();
                // const duration = endTime - startTime;
                // console.log(`Upload completed in ${duration} milliseconds`);

                // if (!response.ok) {
                //     throw new Error('Upload failed');
                // }

                // const data = await response.json();
                // console.log('File uploaded successfully:', data);

                // Step 2: Process with lipforensic model
                const formData = new FormData();
                formData.append('file', selectedFile);

                const response = await fetch('http://localhost:8000/api/models/test', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error('Failed to post video');
                }

                const filePath = response.headers.get('File-Path');
                console.log('File path:', filePath);

                const score = response.headers.get('Score');
                console.log('Score:', score);

                const videoBlob = await response.blob();
                const videoUrl = URL.createObjectURL(videoBlob);
                console.log('Video URL:', videoUrl);

                
                // // Simulate upload
                // await new Promise((resolve) => setTimeout(resolve, 1000));
                // console.log('Simulated upload completed in 1000 milliseconds');
                
                setScore(score || '');
                setVideoUrl(videoUrl);
                onFileUpload(selectedFile);

                setPreview(null);
                setIsPreviewShown(false);
                setSelectedFile(null);
            } catch (error) {
                console.error('Upload error:', error);
            } finally {
                setIsProcessing(false);
            }
        }
    }, [selectedFile, onFileUpload, setVideoUrl, setScore]);

    const handleClear = useCallback(() => {
        setPreview(null);
        setIsPreviewShown(false);
    }, []);

    return (
        <div 
            className={`${styles.dropArea} ${isDragging ? styles.dragging : ''} ${isPreviewShown ? styles.previewShown : ''}`}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
        >
            {isProcessing ? (
                <HStack gap="10">
                    <ProgressCircleRoot size="lg" value={null}>
                        <ProgressCircleRing cap="round" />
                    </ProgressCircleRoot>
                </HStack>
            ) : (
                preview ? (
                    <div className={styles.previewContainer}>
                        {selectedFile && selectedFile.type.startsWith('video/') ? (
                            <video autoPlay loop muted className={styles.preview} src={preview} />
                        ) : (
                            <img className={styles.preview} src={preview} alt="Preview" />
                        )}
                        <div className={styles.buttonContainer}>
                            <button onClick={handleClear} className={styles.clearButton}>Clear</button>
                            <button onClick={handleUpload} className={styles.uploadButton}>Upload</button>
                        </div>
                    </div>
                ) : (
                    <form className={styles.uploadForm}>
                        <div className={styles.uploadIconContainer}>
                            <label className={styles.label} htmlFor="fileElem">
                                <svg 
                                    xmlns="http://www.w3.org/2000/svg" 
                                    fill="#ebebeb" 
                                    className={styles.uploadIcon} 
                                    viewBox="0 0 16 16"
                                >
                                    <path d="M6.002 5.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z"></path>
                                    <path d="M2.002 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2h-12zm12 1a1 1 0 0 1 1 1v6.5l-3.777-1.947a.5.5 0 0 0-.577.093l-3.71 3.71-2.66-1.772a.5.5 0 0 0-.63.062L1.002 12V3a1 1 0 0 1 1-1h12z"></path>
                                </svg>
                            </label>
                        </div>
                        <input 
                            type="file" 
                            id="fileElem" 
                            className={styles.fileInput} 
                            accept="image/*,video/*" 
                            onChange={handleFileChange}
                        />
                        <label className={styles.button} htmlFor="fileElem">Upload Image/Video</label>
                    </form>
                )
            )}
        </div>
    );
};

export default FileUpload;