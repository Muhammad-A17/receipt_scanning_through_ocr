import { useNavigate, useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Card, Button, Loader } from './components';

function useQuery() {
    const { search } = useLocation();
    return new URLSearchParams(search);
}

function ProcessingPage() {
    const navigate = useNavigate();
    const query = useQuery();
    const id = query.get('id');

    const [status, setStatus] = useState('idle');
    const [error, setError] = useState(null);
    const [progress, setProgress] = useState(0);
    const [stage, setStage] = useState('Initializing...');

    useEffect(() => {
        const runProcessing = async () => {
            if (!id) {
                setError('Missing Receipt ID');
                setStatus('error');
                return;
            }
            
            setStatus('processing');
            setError(null);
            setProgress(0);
            setStage('Initializing...');

            // Realistic progress simulation with stages
            const stages = [
                { progress: 10, message: 'Loading image...' },
                { progress: 25, message: 'Preparing OCR models...' },
                { progress: 40, message: 'Extracting text...' },
                { progress: 60, message: 'Processing image variations...' },
                { progress: 75, message: 'Analyzing receipt data...' },
                { progress: 85, message: 'Extracting details...' },
                { progress: 92, message: 'Finalizing...' },
            ];

            let currentStageIndex = 0;
            const progressInterval = setInterval(() => {
                if (currentStageIndex < stages.length) {
                    const stage = stages[currentStageIndex];
                    setProgress(stage.progress);
                    setStage(stage.message);
                    currentStageIndex++;
                } else {
                    // Slow progress from 92% to 98% to show it's still working
                    setProgress(prev => {
                        if (prev < 98) {
                            return prev + 0.5;
                        }
                        return prev;
                    });
                    if (progress >= 92 && progress < 98) {
                        setStage('Almost done...');
                    }
                }
            }, 800); // Slower updates for more realistic feel

            // Slow progress after 98% while waiting for API
            const slowProgressInterval = setInterval(() => {
                setProgress(prev => {
                    if (prev < 99) {
                        return prev + 0.1;
                    }
                    return prev;
                });
            }, 2000);

            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout

                const res = await fetch(`http://127.0.0.1:8000/api/receipts/${id}/process/`, {
                    method: 'POST',
                    signal: controller.signal,
                });

                clearTimeout(timeoutId);
                clearInterval(progressInterval);
                clearInterval(slowProgressInterval);
                setProgress(100);
                setStage('Complete!');

                if (!res.ok) {
                    const errorData = await res.json().catch(() => ({}));
                    throw new Error(errorData.error || `Processing failed: HTTP ${res.status}`);
                }
                
                const data = await res.json();
                console.log('Processing Result:', data);
                
                // Small delay to show 100% before completing
                setTimeout(() => {
                    setStatus('completed');
                }, 500);
            } catch (e) {
                clearInterval(progressInterval);
                clearInterval(slowProgressInterval);
                
                if (e.name === 'AbortError') {
                    setError('Processing timed out. The receipt may be too complex. Please try again.');
                } else {
                    setError(String(e));
                }
                setStatus('error');
            }
        };
        
        runProcessing();
    }, [id]);
    
    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/30 to-gray-50 pt-16 transition-colors duration-300">
            <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <Card className="border-2">
                    <div className="text-center space-y-8">
                        {/* Status Icon */}
                        <div className="flex justify-center">
                            {status === 'processing' && (
                                <div className="space-y-4">
                                    <Loader 
                                        text="Processing" 
                                        words={["receipt", "data", "information", "details", "receipt"]} 
                                    />
                                </div>
                            )}
                            {status === 'completed' && (
                                <div className="relative">
                                    <div className="absolute inset-0 bg-green-500 rounded-full blur-2xl opacity-30"></div>
                                    <div className="relative w-24 h-24 bg-green-600 rounded-full flex items-center justify-center shadow-2xl border-2 border-green-700">
                                        <svg className="w-14 h-14 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={3}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                        </svg>
                                    </div>
                                </div>
                            )}
                            {status === 'error' && (
                                <div className="relative">
                                    <div className="absolute inset-0 bg-red-500 rounded-full blur-2xl opacity-30"></div>
                                    <div className="relative w-24 h-24 bg-red-600 rounded-full flex items-center justify-center shadow-2xl border-2 border-red-700">
                                        <svg className="w-14 h-14 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={3}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                                        </svg>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Status Text */}
                        <div className="space-y-2">
                            <h1 className="text-3xl font-bold text-gray-900">
                                {status === 'processing' && 'Processing Receipt'}
                                {status === 'completed' && 'Processing Complete!'}
                                {status === 'error' && 'Processing Failed'}
                            </h1>
                            <p className="text-lg text-gray-700 font-medium">
                                {status === 'processing' && stage}
                                {status === 'completed' && 'All data has been successfully extracted and saved'}
                                {status === 'error' && 'An error occurred while processing your receipt'}
                            </p>
                        </div>

                        {/* Progress Bar */}
                        {status === 'processing' && (
                            <div className="space-y-3">
                                <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden shadow-inner">
                                    <div 
                                        className="bg-gradient-to-r from-blue-500 to-blue-600 h-full rounded-full transition-all duration-500 shadow-lg"
                                        style={{ width: `${progress}%` }}
                                    ></div>
                                </div>
                                <div className="flex justify-between items-center">
                                    <p className="text-sm font-semibold text-gray-700">
                                        {Math.round(progress)}% Complete
                                    </p>
                                    <p className="text-xs text-gray-500">
                                        {progress < 50 && 'This may take a minute...'}
                                        {progress >= 50 && progress < 90 && 'Almost there...'}
                                        {progress >= 90 && 'Finalizing...'}
                                    </p>
                                </div>
                            </div>
                        )}

                        {/* Error Message */}
                        {error && (
                            <div className="p-4 bg-red-50 border-2 border-red-200 rounded-xl">
                                <p className="text-sm font-semibold text-red-800">{error}</p>
                            </div>
                        )}

                        {/* Action Buttons */}
                        <div className="flex flex-col sm:flex-row gap-3 justify-center pt-4">
                            {status === 'completed' && (
                                <Button
                                    onClick={() => navigate(`/receipt/${id}`)}
                                    size="lg"
                                    className="w-full sm:w-auto min-w-[200px]"
                                >
                                    View Receipt Details
                                </Button>
                            )}
                            {status === 'error' && (
                                <>
                                    <Button
                                        onClick={() => navigate('/upload')}
                                        size="lg"
                                        className="w-full sm:w-auto min-w-[200px]"
                                    >
                                        Try Again
                                    </Button>
                                    <Button
                                        onClick={() => navigate('/receipts')}
                                        color="outline"
                                        size="lg"
                                        className="w-full sm:w-auto min-w-[200px]"
                                    >
                                        View History
                                    </Button>
                                </>
                            )}
                        </div>
                    </div>
                </Card>
            </div>
        </div>
    );
}

export default ProcessingPage;
