import React from 'react';

interface AugmentedVideoProps {
  videoUrl: string;
  score: string | null;
}

const AugmentedVideo: React.FC<AugmentedVideoProps> = ({ videoUrl, score }) => {
  return (
    <div className="bg-gray-100">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto py-8 sm:py-12 lg:py-4">
          <h1 className="text-3xl font-bold mb-4">Augmented Video</h1>

          <div className="mt-6 space-y-12 lg:grid lg:grid-cols-1 lg:gap-x-6 lg:space-y-0">
            <div className="group relative">
              <video
                className="w-full max-w-2xl rounded-lg bg-white object-cover group-hover:opacity-75"
                autoPlay
                loop
                muted
                src={videoUrl}
                style={{ aspectRatio: '16/9' }} // Adjust aspect ratio to match original dimensions
              />
              <h3 className="mt-6 text-sm text-gray-500">Augmented Video Score: {score}</h3> {/* Update this line */}
              </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AugmentedVideo;