import React from 'react';

interface ProcessingPageProps {
  originalVideoSrc: string;
  roiVideos: {
    leftEye: string;
    nose: string;
    mouth: string;
  };
}

const ProcessingPage: React.FC<ProcessingPageProps> = ({ originalVideoSrc, roiVideos }) => {
  const callouts = [
    {
      name: 'Left Eye',
      description: 'ROI video of the left eye.',
      imageSrc: roiVideos.leftEye,
      imageAlt: 'Left eye video preview.',
      href: '#',
    },
    {
      name: 'Nose', 
      description: 'ROI video of the nose.',
      imageSrc: roiVideos.nose,
      imageAlt: 'Nose video preview.',
      href: '#',
    },
    {
      name: 'Mouth',
      description: 'ROI video of the mouth.',
      imageSrc: roiVideos.mouth,
      imageAlt: 'Mouth video preview.',
      href: '#',
    },
  ];

  return (
    <div className="bg-gray-100">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-2xl py-8 sm:py-12 lg:max-w-none lg:py-4">
          <h1 className="text-3xl font-bold mb-4">Processing Results</h1>

          <div className="mb-6">
            <video
              autoPlay
              loop
              muted
              controlsList="nodownload nofullscreen noremoteplayback"
              className="w-full max-w-2xl"
              src={originalVideoSrc}
            />
          </div>

          <h2 className="text-2xl font-bold text-gray-900">Regions of Interest</h2>

          <div className="mt-6 space-y-12 lg:grid lg:grid-cols-3 lg:gap-x-6 lg:space-y-0">
            {callouts.map((callout) => (
              <div key={callout.name} className="group relative">
                <video
                  autoPlay
                  loop
                  muted
                  controlsList="nodownload nofullscreen noremoteplayback"
                  className="w-full rounded-lg bg-white object-cover group-hover:opacity-75 max-sm:h-80 sm:aspect-[2/1] lg:aspect-square"
                  src={callout.imageSrc}
                />
                <h3 className="mt-6 text-sm text-gray-500">
                  {callout.name}
                </h3>
                <p className="text-base font-semibold text-gray-900">{callout.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProcessingPage;