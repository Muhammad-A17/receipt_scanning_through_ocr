import React from 'react';

export function Loader({ text = "Processing", words = ["receipt", "data", "information", "details", "receipt"] }) {
  return (
    <div className="flex items-center justify-center p-8">
      <div className="bg-white rounded-2xl px-8 py-4 border border-gray-200/60 shadow-md backdrop-blur-sm">
        <div className="flex items-center text-gray-600 font-medium text-xl font-sans h-10 py-2.5 px-2.5 rounded-lg">
          <span className="mr-2">{text}</span>
          <div className="relative overflow-hidden h-full" style={{ height: '40px' }}>
            <div className="absolute inset-0 bg-gradient-to-b from-gray-100 via-transparent to-gray-100 z-20 pointer-events-none"></div>
            <div className="relative h-full">
              {words.map((word, index) => (
                <span
                  key={index}
                  className="block h-full pl-1.5 text-blue-600 absolute top-0 left-0 w-full"
                  style={{
                    animation: `spin-words 4s infinite`,
                    animationDelay: `${index * 0.5}s`
                  }}
                >
                  {word}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Loader;
