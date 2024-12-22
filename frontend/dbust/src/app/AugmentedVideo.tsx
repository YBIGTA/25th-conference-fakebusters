import React, { useState, useEffect } from 'react';
import { ProgressCircleRing, ProgressCircleRoot } from "@/components/ui/progress-circle";
import { HStack } from "@chakra-ui/react";

interface AugmentedVideoProps {
  images: string[];
}

const AugmentedVideo: React.FC<AugmentedVideoProps> = ({ images }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isWaiting, setIsWaiting] = useState(false);


  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (currentIndex < images.length - 1) {
      interval = setInterval(() => {
        setCurrentIndex((prevIndex) => prevIndex + 1);
      }, 1000 / 60);
    } else {
      setIsWaiting(true);
      // Simulate receiving new images after a delay
      setTimeout(() => {
        setIsWaiting(false);
        setCurrentIndex(0); // Reset to the beginning when new images are "received"
      }, 0); // 0 second delay
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [currentIndex, images.length]);

  return (
    <div className="bg-gray-100">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-2xl py-8 sm:py-12 lg:max-w-none lg:py-4">
          <h1 className="text-3xl font-bold mb-4">Augmented Video</h1>

          <div className="mt-6 space-y-12 lg:grid lg:grid-cols-1 lg:gap-x-6 lg:space-y-0">
            <div className="group relative">
              <div className="relative">
                <img
                  className="w-full max-w-2xl rounded-lg bg-white object-cover group-hover:opacity-75"
                  src={images[currentIndex]}
                  alt={`Augmented frame ${currentIndex + 1}`}
                  style={{ aspectRatio: '16/9' }}
                />
                {isWaiting && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
                    <HStack gap="10">
                      <ProgressCircleRoot size="lg" value={null}>
                        <ProgressCircleRing cap="round" />
                      </ProgressCircleRoot>
                    </HStack>
                  </div>
                )}
              </div>
              <h3 className="mt-6 text-sm text-gray-500">Frame {currentIndex + 1}</h3>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AugmentedVideo;