import React, { useState, useCallback } from 'react';
import styles from './FileUpload.module.css';
import { ProgressCircleRing, ProgressCircleRoot } from "@/components/ui/progress-circle";
import { HStack } from "@chakra-ui/react";
import { getConfig } from '@/config';

const config = getConfig();

interface FileUploadProps {
    onFileUpload: (file: File) => void;
    lipSetScore: (score: string) => void;
    lipSetVideoUrl: (url: string) => void;
    mmnetSetScore: (score: string) => void;
    mmnetSetVideoUrl: (url: string) => void;
    setPpgVideos: (ppgVideos: any) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, lipSetScore, lipSetVideoUrl, mmnetSetScore, mmnetSetVideoUrl,setPpgVideos }) => {
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

    const uploadLipreading = useCallback(async (file: File) => {
        try {
            const lipFormData = new FormData();
            lipFormData.append('file', file);

            const lipResponse = await fetch(`${(await config).apiBaseUrl}/api/models/lipforensic`, {
                method: 'POST',
                body: lipFormData,
            });

            if (!lipResponse.ok) {
                throw new Error('Failed to post video');
            }

            const lipScore = lipResponse.headers.get('Score');
            const lipVideoBlob = await lipResponse.blob();
            const lipVideoUrl = URL.createObjectURL(lipVideoBlob);

            lipSetScore(lipScore || '');
            lipSetVideoUrl(lipVideoUrl);
        } catch (error) {
            console.error('Lipreading upload error:', error);
        }
    }, [lipSetScore, lipSetVideoUrl]);

    const uploadMMNET = useCallback(async (file: File) => {
        try {
            const mmnetFormData = new FormData();
            mmnetFormData.append('file', file);

            const mmnetResponse = await fetch(`${(await config).apiBaseUrl}/api/models/mmnet`, {
                method: 'POST',
                body: mmnetFormData,
            });

            if (!mmnetResponse.ok) {
                throw new Error('Failed to post video');
            }

            const mmnetScore = mmnetResponse.headers.get('Score');
            const mmnetVideoBlob = await mmnetResponse.blob();
            const mmnetVideoUrl = URL.createObjectURL(mmnetVideoBlob);

            mmnetSetScore(mmnetScore || '');
            mmnetSetVideoUrl(mmnetVideoUrl);
        } catch (error) {
            console.error('MMNET upload error:', error);
        }
    }, [mmnetSetScore, mmnetSetVideoUrl]);

    

    const uploadVisualPPG = useCallback(async (file: File) => {
        try {
            const ppgFormData = new FormData();
            ppgFormData.append('file', file);

            const ppgResponse = await fetch(`${(await config).apiBaseUrl}/api/models/visual-ppg`, {
                method: 'POST',
                body: ppgFormData,
            });

            if (!ppgResponse.ok) {
                throw new Error('Failed to post video');
            }

            const ppgData = await ppgResponse.json();
            console.log('Visual PPG response:', ppgData);

            // Store the video names
            return ppgData.videos;
        } catch (error) {
            console.error('Visual PPG upload error:', error);
            return null;
        }
    }, []);

    const getPPGVideo = useCallback(async (videoNames: { ppg_graph: string; ppg_mask: string; ppg_transformed: string }) => {
        try {
            const header = new Headers();
            header.append('ngrok-skip-browser-warning', 'true');
            const graphBaseUrl = (await config).apiBaseUrl;
            const ppgGraphResponse = await fetch(`${graphBaseUrl}/api/models/video/${videoNames.ppg_graph}` , { headers: header });
            const ppgMaskResponse = await fetch(`${graphBaseUrl}/api/models/video/${videoNames.ppg_mask}` , { headers: header });
            const ppgTransformedResponse = await fetch(`${graphBaseUrl}/api/models/video/${videoNames.ppg_transformed}` , { headers: header });

            if (!ppgGraphResponse.ok || !ppgMaskResponse.ok || !ppgTransformedResponse.ok) {
                throw new Error('Failed to fetch PPG videos');
            }

            const ppgGraphBlob = await ppgGraphResponse.blob();
            const ppgMaskBlob = await ppgMaskResponse.blob();
            const ppgTransformedBlob = await ppgTransformedResponse.blob();

            console.log(`${graphBaseUrl}${videoNames.ppg_graph}`)
            console.log(ppgMaskBlob)

            const ppgGraphUrl = URL.createObjectURL(ppgGraphBlob);
            const ppgMaskUrl = URL.createObjectURL(ppgMaskBlob);
            const ppgTransformedUrl = URL.createObjectURL(ppgTransformedBlob);

            return {
                ppgGraphUrl,
                ppgMaskUrl,
                ppgTransformedUrl,
            };
        } catch (error) {
            console.error('Failed to fetch PPG videos:', error);
            return null;
        }
    }, []);


    const handleUpload = useCallback(async () => {
        if (selectedFile) {
            setIsProcessing(true);

            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const [mmnetResult, lipreadingResult, visualPpgResult] = await Promise.all([
                uploadMMNET(selectedFile),
                uploadLipreading(selectedFile),
                uploadVisualPPG(selectedFile)
            ]);
    
            if (visualPpgResult) {
                const ppgVideoUrls = await getPPGVideo(visualPpgResult);
                setPpgVideos(ppgVideoUrls);
            }

            onFileUpload(selectedFile);

            setPreview(null);
            setIsPreviewShown(false);
            setSelectedFile(null);
            setIsProcessing(false);
        }
    }, [selectedFile, onFileUpload, uploadLipreading, uploadMMNET, uploadVisualPPG]);

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
                <div className={styles.previewContainer}>
                    <HStack gap="10" className={styles.progressContainer}>
                        <ProgressCircleRoot size="lg" value={null}>
                            <ProgressCircleRing cap="round" />
                        </ProgressCircleRoot>
                    </HStack>
                    {selectedFile && selectedFile.type.startsWith('video/') ? (
                        <video autoPlay loop muted className={`${styles.preview} ${styles.processing}`} src={preview || undefined} />
                    ) : (
                        <img className={`${styles.preview} ${styles.processing}`} src={preview || undefined} alt="Preview" />
                    )}
                </div>
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