import { useNavigate } from 'react-router-dom';
import { useState, useRef } from 'react';
import { Card, Button } from './components';

function UploadPage() {
    const navigate = useNavigate();
    const fileInputRef = useRef(null);
    
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);
    const [receiptId, setReceiptId] = useState(null);
    const [dragActive, setDragActive] = useState(false);

    const onFileChange = (selectedFile) => {
        if (!selectedFile) return;
        
        setFile(selectedFile);
        setError(null);
        
        const reader = new FileReader();
        reader.onload = (e) => {
            setImagePreview(e.target.result);
        };
        reader.readAsDataURL(selectedFile);
    };

    const handleFileInput = (e) => {
        const selectedFile = e.target.files?.[0] ?? null;
        onFileChange(selectedFile);
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            onFileChange(e.dataTransfer.files[0]);
        }
    };

    const onUpload = async () => {
        if (!file) {
            setError('Please select an image first');
            return;
        }

        setUploading(true);
        setError(null);
        try {
            const form = new FormData();
            form.append('image', file);

            const res = await fetch('http://127.0.0.1:8000/api/receipts/upload/', {
                method: 'POST',
                body: form,
            });
            if (!res.ok) throw new Error(`Upload failed: HTTP ${res.status}`);

            const data = await res.json();
            setReceiptId(data.id);
        } catch (e) {
            setError(String(e));
        } finally {
            setUploading(false);
        }
    };

    const goToProcessing = () => {
        if (receiptId) navigate(`/processing?id=${receiptId}`);
    };
    
    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/30 to-gray-50 pt-16 transition-colors duration-300">
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
                        Upload Receipt
                    </h1>
                    <p className="text-lg text-gray-700 dark:text-gray-200 font-medium">
                        Upload an image of your receipt to extract data using AI
                    </p>
                </div>

                <Card className="border-2">
                    <div className="space-y-6">
                        {/* File Upload Area */}
                        <div
                            onDragEnter={handleDrag}
                            onDragLeave={handleDrag}
                            onDragOver={handleDrag}
                            onDrop={handleDrop}
                            className={`
                                relative border-2 border-dashed rounded-2xl p-16 text-center transition-all duration-300
                                ${dragActive 
                                    ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-900/20 scale-[1.02]' 
                                    : 'border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500'
                                }
                                ${imagePreview ? 'bg-gray-50' : 'bg-white'}
                            `}
                        >
                            {!imagePreview ? (
                                <div className="flex flex-col items-center space-y-6">
                                    <div className="relative">
                                        <div className="absolute inset-0 bg-blue-500 rounded-full blur-2xl opacity-20"></div>
                                        <div className="relative w-24 h-24 bg-blue-600 dark:bg-blue-500 rounded-2xl flex items-center justify-center shadow-xl border-2 border-blue-700 dark:border-blue-400">
                                            <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                            </svg>
                                        </div>
                                    </div>
                                    <div>
                                        <p className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                                            Drag and drop your receipt image
                                        </p>
                                        <p className="text-sm text-gray-600 dark:text-gray-300 font-medium">
                                            or click the button below to browse
                                        </p>
                                    </div>
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept="image/*"
                                        onChange={handleFileInput}
                                        className="hidden"
                                    />
                                    <Button
                                        onClick={() => fileInputRef.current?.click()}
                                        size="lg"
                                    >
                                        Choose File
                                    </Button>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    <div className="relative inline-block group">
                                        <img 
                                            src={imagePreview} 
                                            alt="Receipt preview" 
                                            className="max-h-80 rounded-xl shadow-2xl border-4 border-white dark:border-gray-700"
                                        />
                                        <button
                                            onClick={() => {
                                                setFile(null);
                                                setImagePreview(null);
                                                setReceiptId(null);
                                                if (fileInputRef.current) fileInputRef.current.value = '';
                                            }}
                                            className="absolute top-4 right-4 p-2 bg-red-600 text-white rounded-full hover:bg-red-700 transition-all shadow-lg hover:scale-110"
                                        >
                                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                            </svg>
                                        </button>
                                    </div>
                                    <div className="flex items-center justify-center space-x-2 text-sm text-gray-700 dark:text-gray-200">
                                        <svg className="w-5 h-5 text-gray-600 dark:text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                        </svg>
                                        <span className="font-semibold">{file?.name}</span>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Error Message */}
                        {error && (
                            <div className="p-4 bg-red-50 dark:bg-red-900/20 border-2 border-red-200 dark:border-red-800 rounded-xl">
                                <div className="flex items-center space-x-2">
                                    <svg className="w-5 h-5 text-red-700 dark:text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    <p className="text-sm font-semibold text-red-800 dark:text-red-300">{error}</p>
                                </div>
                            </div>
                        )}

                        {/* Success Message */}
                        {receiptId && (
                            <div className="p-4 bg-green-50 dark:bg-green-900/20 border-2 border-green-200 dark:border-green-800 rounded-xl">
                                <div className="flex items-center space-x-2">
                                    <svg className="w-5 h-5 text-green-700 dark:text-green-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    <p className="text-sm font-semibold text-green-800 dark:text-green-300">
                                        Upload successful! Ready to process.
                                    </p>
                                </div>
                            </div>
                        )}

                        {/* Action Buttons */}
                        <div className="flex flex-col sm:flex-row gap-3">
                            <Button
                                onClick={onUpload}
                                disabled={!file || uploading || receiptId}
                                className="flex-1"
                                size="lg"
                            >
                                {uploading ? (
                                    <span className="flex items-center space-x-2">
                                        <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        <span>Uploading...</span>
                                    </span>
                                ) : receiptId ? (
                                    <span className="flex items-center space-x-2">
                                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                                        </svg>
                                        <span>Uploaded</span>
                                    </span>
                                ) : (
                                    'Upload Image'
                                )}
                            </Button>
                            {receiptId && (
                                <Button
                                    onClick={goToProcessing}
                                    color="success"
                                    className="flex-1"
                                    size="lg"
                                >
                                    Process Receipt
                                </Button>
                            )}
                        </div>
                    </div>
                </Card>
            </div>
        </div>
    );
}

export default UploadPage;
