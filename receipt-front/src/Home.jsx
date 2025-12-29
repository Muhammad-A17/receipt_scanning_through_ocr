import { useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Card, Button } from './components';

function Home(){
    const navigate = useNavigate();
    const [stats, setStats] = useState({ total: 0, processed: 0, totalSpent: 0 });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const res = await fetch('http://127.0.0.1:8000/api/receipts/');
                if (res.ok) {
                    const receipts = await res.json();
                    const processed = receipts.filter(r => r.processed);
                    const totalSpent = processed.reduce((sum, r) => sum + (parseFloat(r.total) || 0), 0);
                    setStats({
                        total: receipts.length,
                        processed: processed.length,
                        totalSpent: totalSpent.toFixed(2)
                    });
                }
            } catch (err) {
                console.error('Error fetching stats:', err);
            } finally {
                setLoading(false);
            }
        };
        fetchStats();
    }, []);

    return(
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/30 to-gray-50 pt-16 transition-colors duration-300">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <div className="space-y-12">
                    {/* Hero Section */}
                    <div className="text-center space-y-6">
                        <div className="inline-flex items-center justify-center w-24 h-24 bg-blue-600 dark:bg-blue-500 rounded-3xl shadow-2xl shadow-blue-500/40 dark:shadow-blue-500/30 mb-6 transform hover:scale-105 transition-transform">
                            <svg className="w-14 h-14 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                        </div>
                        <h1 className="text-5xl sm:text-6xl font-extrabold text-gray-900 dark:text-white">
                            Receipt Scanner
                        </h1>
                        <p className="text-xl text-gray-700 dark:text-gray-200 max-w-2xl mx-auto leading-relaxed font-medium">
                            Transform your receipts into structured data with AI-powered OCR technology. 
                            Extract merchant details, items, totals, and more in seconds.
                        </p>
                    </div>

                    {/* Stats Cards */}
                    {!loading && (
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <Card className="text-center border-2 border-blue-100 dark:border-blue-900 hover:border-blue-300 dark:hover:border-blue-700 transition-colors">
                                <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 dark:bg-blue-900/50 rounded-2xl mb-4 border-2 border-blue-200 dark:border-blue-800">
                                    <svg className="w-8 h-8 text-blue-700 dark:text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                </div>
                                <div className="text-4xl font-bold text-blue-700 dark:text-blue-300 mb-2">
                                    {stats.total}
                                </div>
                                <div className="text-sm font-semibold text-gray-700 dark:text-gray-200">Total Receipts</div>
                            </Card>
                            <Card className="text-center border-2 border-green-100 dark:border-green-900 hover:border-green-300 dark:hover:border-green-700 transition-colors">
                                <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 dark:bg-green-900/50 rounded-2xl mb-4 border-2 border-green-200 dark:border-green-800">
                                    <svg className="w-8 h-8 text-green-700 dark:text-green-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                </div>
                                <div className="text-4xl font-bold text-green-700 dark:text-green-300 mb-2">
                                    {stats.processed}
                                </div>
                                <div className="text-sm font-semibold text-gray-700 dark:text-gray-200">Processed</div>
                            </Card>
                            <Card className="text-center border-2 border-purple-100 dark:border-purple-900 hover:border-purple-300 dark:hover:border-purple-700 transition-colors">
                                <div className="inline-flex items-center justify-center w-16 h-16 bg-purple-100 dark:bg-purple-900/50 rounded-2xl mb-4 border-2 border-purple-200 dark:border-purple-800">
                                    <svg className="w-8 h-8 text-purple-700 dark:text-purple-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                </div>
                                <div className="text-4xl font-bold text-purple-700 dark:text-purple-300 mb-2">
                                    ${stats.totalSpent}
                                </div>
                                <div className="text-sm font-semibold text-gray-700 dark:text-gray-200">Total Spent</div>
                            </Card>
                        </div>
                    )}

                    {/* Action Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <Card className="group hover:shadow-2xl transition-all duration-300 border-2 border-transparent hover:border-blue-200 dark:hover:border-blue-800">
                            <div className="flex items-start space-x-6 mb-6">
                                <div className="w-16 h-16 bg-blue-600 dark:bg-blue-500 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/40 dark:shadow-blue-500/30 group-hover:scale-110 transition-transform border-2 border-blue-700 dark:border-blue-400">
                                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                    </svg>
                                </div>
                                <div className="flex-1">
                                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Upload Receipt</h3>
                                    <p className="text-gray-700 dark:text-gray-200 leading-relaxed font-medium">
                                        Upload a receipt image and let our AI extract all the important information automatically.
                                    </p>
                                </div>
                            </div>
                            <Button onClick={() => navigate('/upload')} className="w-full" size="lg">
                                Upload Receipt
                            </Button>
                        </Card>

                        <Card className="group hover:shadow-2xl transition-all duration-300 border-2 border-transparent hover:border-green-200 dark:hover:border-green-800">
                            <div className="flex items-start space-x-6 mb-6">
                                <div className="w-16 h-16 bg-green-600 dark:bg-green-500 rounded-2xl flex items-center justify-center shadow-lg shadow-green-500/40 dark:shadow-green-500/30 group-hover:scale-110 transition-transform border-2 border-green-700 dark:border-green-400">
                                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                </div>
                                <div className="flex-1">
                                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">View History</h3>
                                    <p className="text-gray-700 dark:text-gray-200 leading-relaxed font-medium">
                                        Browse all your processed receipts, search, filter, and manage your expense records.
                                    </p>
                                </div>
                            </div>
                            <Button onClick={() => navigate('/receipts')} color="success" className="w-full" size="lg">
                                View History
                            </Button>
                        </Card>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Home;
