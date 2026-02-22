import { useNavigate, useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Card, Button } from './components';
import api from './services/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace('/api', '') || 'http://localhost:8000';

function ReceiptDetailPage() {
    const navigate = useNavigate();
    const { id } = useParams();

    const [receipt, setReceipt] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const loadReceipt = async () => {
            try {
                const data = await api.getReceipt(id);
                setReceipt(data);
            } catch (e) {
                setError(String(e));
            } finally {
                setLoading(false);
            }
        };

        loadReceipt();
    }, [id]);

    const formatDate = (dateString) => {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    };

    const formatTime = (timeString) => {
        if (!timeString) return 'N/A';
        return timeString;
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/30 to-gray-50 pt-16 transition-colors duration-300">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                    <Card className="border-2">
                        <div className="text-center py-16">
                            <div className="inline-block w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
                            <p className="text-gray-600 font-medium">Loading receipt...</p>
                        </div>
                    </Card>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/30 to-gray-50 pt-16 transition-colors duration-300">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                    <Card className="border-2">
                        <div className="text-center py-16 space-y-6">
                            <div className="relative inline-block">
                                <div className="absolute inset-0 bg-red-500 rounded-full blur-2xl opacity-30"></div>
                                <div className="relative w-20 h-20 bg-red-600 rounded-full flex items-center justify-center shadow-xl border-2 border-red-700">
                                    <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </div>
                            </div>
                            <div>
                                <h2 className="text-2xl font-bold text-gray-900 mb-2">Error Loading Receipt</h2>
                                <p className="text-red-700 font-semibold">{error}</p>
                            </div>
                            <Button onClick={() => navigate('/receipts')} size="lg">
                                Back to History
                            </Button>
                        </div>
                    </Card>
                </div>
            </div>
        );
    }

    if (!receipt) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/30 to-gray-50 pt-16 transition-colors duration-300">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                    <Card className="border-2">
                        <div className="text-center py-16">
                            <p className="text-gray-600 mb-6">Receipt not found</p>
                            <Button onClick={() => navigate('/receipts')} size="lg">
                                Back to History
                            </Button>
                        </div>
                    </Card>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-white pt-16 transition-colors duration-300">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <div className="space-y-8">
                    {/* Header */}
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                        <div>
                            <h1 className="text-4xl font-bold text-gray-900 mb-2">
                                Receipt Details
                            </h1>
                            <p className="text-lg text-gray-700 font-medium">
                                {receipt.merchant_name || 'Unknown Merchant'}
                            </p>
                        </div>
                        <div className="flex gap-3">
                            <Button onClick={() => navigate('/receipts')} color="outline" size="md" className="flex items-center space-x-2">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                                </svg>
                                <span>Back to History</span>
                            </Button>
                            <Button onClick={() => navigate('/')} size="md" className="flex items-center space-x-2">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                                </svg>
                                <span>Home</span>
                            </Button>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Main Content */}
                        <div className="lg:col-span-2 space-y-6">
                            {/* Merchant Information */}
                            <Card title="Merchant Information" className="border-2">
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                                    <div className="flex items-start space-x-3">
                                        <div className="flex-shrink-0 w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center border border-blue-200">
                                            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                                            </svg>
                                        </div>
                                        <div className="flex-1">
                                            <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Name</span>
                                            <p className="text-lg font-bold text-gray-900 mt-1">
                                                {receipt.merchant_name || 'N/A'}
                                            </p>
                                        </div>
                                    </div>
                                    {receipt.merchant_phone && (
                                        <div className="flex items-start space-x-3">
                                            <div className="flex-shrink-0 w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center border border-green-200">
                                                <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                                                </svg>
                                            </div>
                                            <div className="flex-1">
                                                <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Phone</span>
                                                <p className="text-lg font-bold text-gray-900 mt-1">
                                                    {receipt.merchant_phone}
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                    {receipt.merchant_email && (
                                        <div className="flex items-start space-x-3">
                                            <div className="flex-shrink-0 w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center border border-purple-200">
                                                <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                                </svg>
                                            </div>
                                            <div className="flex-1">
                                                <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Email</span>
                                                <p className="text-lg font-bold text-gray-900 mt-1 break-all">
                                                    {receipt.merchant_email}
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                    {receipt.merchant_address && (
                                        <div className="sm:col-span-2 flex items-start space-x-3">
                                            <div className="flex-shrink-0 w-10 h-10 bg-orange-100 rounded-lg flex items-center justify-center border border-orange-200">
                                                <svg className="w-5 h-5 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                                                </svg>
                                            </div>
                                            <div className="flex-1">
                                                <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Address</span>
                                                <p className="text-lg font-bold text-gray-900 mt-1">
                                                    {receipt.merchant_address}
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </Card>

                            {/* Transaction Information */}
                            <Card title="Transaction Information" className="border-2">
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                                    <div className="flex items-start space-x-3">
                                        <div className="flex-shrink-0 w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center border border-indigo-200">
                                            <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                            </svg>
                                        </div>
                                        <div className="flex-1">
                                            <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Date</span>
                                            <p className="text-lg font-bold text-gray-900 mt-1">
                                                {formatDate(receipt.date)}
                                            </p>
                                        </div>
                                    </div>
                                    {receipt.time && (
                                        <div className="flex items-start space-x-3">
                                            <div className="flex-shrink-0 w-10 h-10 bg-teal-100 rounded-lg flex items-center justify-center border border-teal-200">
                                                <svg className="w-5 h-5 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                </svg>
                                            </div>
                                            <div className="flex-1">
                                                <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Time</span>
                                                <p className="text-lg font-bold text-gray-900 mt-1">
                                                    {formatTime(receipt.time)}
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                    {receipt.receipt_number && (
                                        <div className="flex items-start space-x-3">
                                            <div className="flex-shrink-0 w-10 h-10 bg-pink-100 rounded-lg flex items-center justify-center border border-pink-200">
                                                <svg className="w-5 h-5 text-pink-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                                </svg>
                                            </div>
                                            <div className="flex-1">
                                                <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Receipt Number</span>
                                                <p className="text-lg font-bold text-gray-900 mt-1">
                                                    {receipt.receipt_number}
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                    {receipt.transaction_id && (
                                        <div className="flex items-start space-x-3">
                                            <div className="flex-shrink-0 w-10 h-10 bg-cyan-100 rounded-lg flex items-center justify-center border border-cyan-200">
                                                <svg className="w-5 h-5 text-cyan-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                                                </svg>
                                            </div>
                                            <div className="flex-1">
                                                <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Transaction ID</span>
                                                <p className="text-lg font-bold text-gray-900 mt-1 break-all">
                                                    {receipt.transaction_id}
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </Card>

                            {/* Items */}
                            {receipt.items && receipt.items.length > 0 && (
                                <Card title="Items" className="border-2">
                                    <div className="space-y-3">
                                        {receipt.items.map((item, index) => (
                                            <div
                                                key={index}
                                                className="flex justify-between items-start p-4 bg-white rounded-xl border border-gray-200 hover:border-blue-300 transition-colors shadow-sm"
                                            >
                                                <div className="flex items-start space-x-3 flex-1">
                                                    <div className="flex-shrink-0 w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center border border-blue-200">
                                                        <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                            <path strokeLinecap="round" strokeLinejoin="round" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
                                                        </svg>
                                                    </div>
                                                    <div className="flex-1">
                                                        <p className="font-semibold text-gray-900 text-lg">
                                                            {item.name || 'Unknown Item'}
                                                        </p>
                                                        {item.quantity && (
                                                            <div className="flex items-center space-x-2 mt-1">
                                                                <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                                                                </svg>
                                                                <p className="text-sm text-gray-500">
                                                                    Quantity: {item.quantity}
                                                                </p>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                                {item.price && (
                                                    <div className="flex items-center space-x-2">
                                                        <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                                                            <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                        </svg>
                                                        <p className="font-bold text-xl text-gray-900">
                                                            ${parseFloat(item.price).toFixed(2)}
                                                        </p>
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </Card>
                            )}
                        </div>

                        {/* Sidebar */}
                        <div className="space-y-6">
                            {/* Financial Summary */}
                            <Card title="Financial Summary" className="border-2">
                                <div className="space-y-4">
                                    {receipt.sub_total && (
                                        <div className="flex justify-between items-center py-3 border-b border-gray-200">
                                            <div className="flex items-center space-x-2">
                                                <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                                                </svg>
                                                <span className="text-gray-700 font-semibold">Subtotal</span>
                                            </div>
                                            <span className="font-bold text-gray-900 text-lg">
                                                ${parseFloat(receipt.sub_total).toFixed(2)}
                                            </span>
                                        </div>
                                    )}
                                    {receipt.tax && (
                                        <div className="flex justify-between items-center py-3 border-b border-gray-200">
                                            <div className="flex items-center space-x-2">
                                                <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                </svg>
                                                <span className="text-gray-700 font-semibold">Tax</span>
                                            </div>
                                            <span className="font-bold text-gray-900 text-lg">
                                                ${parseFloat(receipt.tax).toFixed(2)}
                                            </span>
                                        </div>
                                    )}
                                    {receipt.tip && (
                                        <div className="flex justify-between items-center py-3 border-b border-gray-200">
                                            <div className="flex items-center space-x-2">
                                                <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                                                </svg>
                                                <span className="text-gray-700 font-semibold">Tip</span>
                                            </div>
                                            <span className="font-bold text-gray-900 text-lg">
                                                ${parseFloat(receipt.tip).toFixed(2)}
                                            </span>
                                        </div>
                                    )}
                                    {receipt.discount && (
                                        <div className="flex justify-between items-center py-3 border-b border-gray-200">
                                            <div className="flex items-center space-x-2">
                                                <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                </svg>
                                                <span className="text-gray-700 font-semibold">Discount</span>
                                            </div>
                                            <span className="font-bold text-green-700 text-lg">
                                                -${parseFloat(receipt.discount).toFixed(2)}
                                            </span>
                                        </div>
                                    )}
                                    <div className="pt-4">
                                        <div className="flex justify-between items-center p-4 bg-blue-50 rounded-xl border-2 border-blue-200">
                                            <div className="flex items-center space-x-2">
                                                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                </svg>
                                                <span className="text-xl font-bold text-gray-900">Total</span>
                                            </div>
                                            <span className="text-3xl font-extrabold text-blue-700">
                                                ${receipt.total ? parseFloat(receipt.total).toFixed(2) : 'N/A'}
                                            </span>
                                        </div>
                                    </div>
                                    {receipt.currency && (
                                        <div className="flex items-center justify-center space-x-2 pt-2">
                                            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                            </svg>
                                            <p className="text-sm text-gray-500 font-medium">
                                                Currency: {receipt.currency}
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </Card>

                            {/* Payment Information */}
                            {(receipt.payment_method || receipt.card_type || receipt.card_last_four) && (
                                <Card title="Payment Information" className="border-2">
                                    <div className="space-y-4">
                                        {receipt.payment_method && (
                                            <div className="flex items-start space-x-3">
                                                <div className="flex-shrink-0 w-10 h-10 bg-emerald-100 rounded-lg flex items-center justify-center border border-emerald-200">
                                                    <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
                                                    </svg>
                                                </div>
                                                <div className="flex-1">
                                                    <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Method</span>
                                                    <p className="text-lg font-bold text-gray-900 mt-1 capitalize">
                                                        {receipt.payment_method}
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                        {receipt.card_type && (
                                            <div className="flex items-start space-x-3">
                                                <div className="flex-shrink-0 w-10 h-10 bg-violet-100 rounded-lg flex items-center justify-center border border-violet-200">
                                                    <svg className="w-5 h-5 text-violet-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
                                                    </svg>
                                                </div>
                                                <div className="flex-1">
                                                    <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Card Type</span>
                                                    <p className="text-lg font-bold text-gray-900 mt-1 capitalize">
                                                        {receipt.card_type}
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                        {receipt.card_last_four && (
                                            <div className="flex items-start space-x-3">
                                                <div className="flex-shrink-0 w-10 h-10 bg-amber-100 rounded-lg flex items-center justify-center border border-amber-200">
                                                    <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                                    </svg>
                                                </div>
                                                <div className="flex-1">
                                                    <span className="text-xs font-bold text-gray-600 uppercase tracking-wide">Card Last Four</span>
                                                    <p className="text-lg font-bold text-gray-900 mt-1 font-mono">
                                                        ****{receipt.card_last_four}
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </Card>
                            )}

                            {/* Receipt Image */}
                            {receipt.image && (
                                <Card title="Receipt Image" className="border-2">
                                    <div className="relative group">
                                        <div className="absolute inset-0 bg-blue-500/10 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
                                        <img
                                            src={`${API_BASE_URL}${receipt.image}`}
                                            alt="Receipt"
                                            className="w-full rounded-xl border-2 border-gray-200 shadow-lg transition-transform group-hover:scale-[1.02]"
                                        />
                                        <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg p-2 shadow-lg">
                                            <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                            </svg>
                                        </div>
                                    </div>
                                </Card>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default ReceiptDetailPage;
