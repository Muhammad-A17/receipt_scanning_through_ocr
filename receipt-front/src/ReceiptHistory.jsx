import { useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Card, Button } from './components';
import api from './services/api';

function ReceiptHistory() {
    const navigate = useNavigate();
    const [receipts, setReceipts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [filterProcessed, setFilterProcessed] = useState('all');
    const [sortBy, setSortBy] = useState('date-desc');
    const [viewMode, setViewMode] = useState('card');
    const [selectedReceipts, setSelectedReceipts] = useState(new Set());
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [deleteTarget, setDeleteTarget] = useState(null); // null for bulk, id for single
    const [deleting, setDeleting] = useState(false);
    const [showSuccessModal, setShowSuccessModal] = useState(false);
    const [showErrorModal, setShowErrorModal] = useState(false);
    const [modalMessage, setModalMessage] = useState('');

    const loadReceipts = async () => {
        try {
            const raw = await api.getReceipts();
            const items = Array.isArray(raw) ? raw : (raw.results ?? []);
            setReceipts(items);
        } catch (err) {
            console.error('Error fetching receipts:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadReceipts();
    }, []);

    const filteredReceipts = receipts.filter(receipt => {
        const matchesSearch = !searchTerm ||
            (receipt.merchant_name?.toLowerCase().includes(searchTerm.toLowerCase())) ||
            (receipt.total?.toString().includes(searchTerm));

        const matchesFilter = filterProcessed === 'all' ||
            (filterProcessed === 'processed' && receipt.processed) ||
            (filterProcessed === 'unprocessed' && !receipt.processed);

        return matchesSearch && matchesFilter;
    });

    const sortedReceipts = [...filteredReceipts].sort((a, b) => {
        switch (sortBy) {
            case 'date-desc':
                return new Date(b.date || b.created_at) - new Date(a.date || a.created_at);
            case 'date-asc':
                return new Date(a.date || a.created_at) - new Date(b.date || b.created_at);
            case 'total-desc':
                return (parseFloat(b.total) || 0) - (parseFloat(a.total) || 0);
            case 'total-asc':
                return (parseFloat(a.total) || 0) - (parseFloat(b.total) || 0);
            case 'merchant-asc':
                return (a.merchant_name || '').localeCompare(b.merchant_name || '');
            case 'merchant-desc':
                return (b.merchant_name || '').localeCompare(a.merchant_name || '');
            default:
                return 0;
        }
    });

    const formatDate = (dateString) => {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    };

    const formatDateTime = (dateString) => {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const handleSelectReceipt = (receiptId) => {
        const newSelected = new Set(selectedReceipts);
        if (newSelected.has(receiptId)) {
            newSelected.delete(receiptId);
        } else {
            newSelected.add(receiptId);
        }
        setSelectedReceipts(newSelected);
    };

    const handleSelectAll = () => {
        if (selectedReceipts.size === sortedReceipts.length) {
            setSelectedReceipts(new Set());
        } else {
            setSelectedReceipts(new Set(sortedReceipts.map(r => r.id)));
        }
    };

    const handleDeleteClick = (receiptId = null) => {
        setDeleteTarget(receiptId);
        setShowDeleteConfirm(true);
    };

    const handleDeleteConfirm = async () => {
        setDeleting(true);
        try {
            if (deleteTarget === null) {
                // Bulk delete
                const ids = Array.from(selectedReceipts);
                const data = await api.bulkDelete(ids);
                setModalMessage(`Successfully deleted ${data.deleted_count} receipt(s)`);
                setShowSuccessModal(true);
                setSelectedReceipts(new Set());
            } else {
                // Single delete
                await api.deleteReceipt(deleteTarget);
            }

            await loadReceipts();
            setShowDeleteConfirm(false);
            setDeleteTarget(null);
        } catch (err) {
            setModalMessage(`Error: ${err.message}`);
            setShowErrorModal(true);
        } finally {
            setDeleting(false);
        }
    };

    const handleExport = (format = 'csv') => {
        const receiptsToExport = deleteTarget === null
            ? sortedReceipts.filter(r => selectedReceipts.has(r.id))
            : sortedReceipts.filter(r => r.id === deleteTarget);

        if (receiptsToExport.length === 0) {
            setModalMessage('Please select at least one receipt to export');
            setShowErrorModal(true);
            return;
        }

        if (format === 'csv') {
            // CSV Export
            const headers = ['ID', 'Merchant Name', 'Date', 'Time', 'Total', 'Subtotal', 'Tax', 'Tip', 'Discount', 'Status', 'Receipt Number'];
            const rows = receiptsToExport.map(r => [
                r.id,
                r.merchant_name || 'N/A',
                r.date || 'N/A',
                r.time || 'N/A',
                r.total || '0.00',
                r.sub_total || '0.00',
                r.tax || '0.00',
                r.tip || '0.00',
                r.discount || '0.00',
                r.processed ? 'Processed' : 'Pending',
                r.receipt_number || 'N/A'
            ]);

            const csvContent = [
                headers.join(','),
                ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
            ].join('\n');

            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `receipts_export_${new Date().toISOString().split('T')[0]}.csv`;
            a.click();
            window.URL.revokeObjectURL(url);
            setModalMessage(`Successfully exported ${receiptsToExport.length} receipt(s) to CSV`);
            setShowSuccessModal(true);
        } else {
            // JSON Export
            const jsonContent = JSON.stringify(receiptsToExport, null, 2);
            const blob = new Blob([jsonContent], { type: 'application/json' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `receipts_export_${new Date().toISOString().split('T')[0]}.json`;
            a.click();
            window.URL.revokeObjectURL(url);
            setModalMessage(`Successfully exported ${receiptsToExport.length} receipt(s) to JSON`);
            setShowSuccessModal(true);
        }
    };

    const stats = {
        total: receipts.length,
        processed: receipts.filter(r => r.processed).length,
        totalSpent: receipts.filter(r => r.processed).reduce((sum, r) => sum + (parseFloat(r.total) || 0), 0).toFixed(2),
        average: receipts.filter(r => r.processed).length > 0
            ? (receipts.filter(r => r.processed).reduce((sum, r) => sum + (parseFloat(r.total) || 0), 0) / receipts.filter(r => r.processed).length).toFixed(2)
            : '0.00'
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/30 to-gray-50 pt-16 transition-colors duration-300">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-4xl font-bold text-gray-900 mb-2">
                        Receipt History
                    </h1>
                    <p className="text-lg text-gray-700 font-medium">
                        View and manage all your processed receipts
                    </p>
                </div>

                {/* Stats Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                    <Card className="border-2 border-blue-200">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-semibold text-gray-600 mb-1">Total Receipts</p>
                                <p className="text-3xl font-bold text-blue-700">{stats.total}</p>
                            </div>
                            <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                            </div>
                        </div>
                    </Card>
                    <Card className="border-2 border-green-200">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-semibold text-gray-600 mb-1">Processed</p>
                                <p className="text-3xl font-bold text-green-700">{stats.processed}</p>
                            </div>
                            <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                        </div>
                    </Card>
                    <Card className="border-2 border-purple-200">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-semibold text-gray-600 mb-1">Total Spent</p>
                                <p className="text-3xl font-bold text-purple-700">${stats.totalSpent}</p>
                            </div>
                            <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center">
                                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                        </div>
                    </Card>
                    <Card className="border-2 border-orange-200">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm font-semibold text-gray-600 mb-1">Average</p>
                                <p className="text-3xl font-bold text-orange-700">${stats.average}</p>
                            </div>
                            <div className="w-12 h-12 bg-orange-100 rounded-xl flex items-center justify-center">
                                <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                            </div>
                        </div>
                    </Card>
                </div>

                <Card className="border-2">
                    {/* Bulk Actions Bar */}
                    {selectedReceipts.size > 0 && (
                        <div className="mb-6 p-4 bg-blue-50 border-2 border-blue-200 rounded-xl flex flex-wrap items-center justify-between gap-4">
                            <div className="flex items-center space-x-4">
                                <span className="font-semibold text-gray-900">
                                    {selectedReceipts.size} receipt{selectedReceipts.size !== 1 ? 's' : ''} selected
                                </span>
                            </div>
                            <div className="flex items-center space-x-2">
                                <Button
                                    onClick={() => handleExport('csv')}
                                    size="sm"
                                    className="bg-green-600 hover:bg-green-700"
                                >
                                    <svg className="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    Export CSV
                                </Button>
                                <Button
                                    onClick={() => handleExport('json')}
                                    size="sm"
                                    className="bg-indigo-600 hover:bg-indigo-700"
                                >
                                    <svg className="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    Export JSON
                                </Button>
                                <Button
                                    onClick={() => handleDeleteClick(null)}
                                    size="sm"
                                    color="danger"
                                >
                                    <svg className="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                    Delete Selected
                                </Button>
                            </div>
                        </div>
                    )}

                    {/* Filters and Controls */}
                    <div className="mb-8 space-y-4">
                        <div className="flex flex-col lg:flex-row gap-4">
                            {/* Search */}
                            <div className="flex-1 relative">
                                <svg className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                </svg>
                                <input
                                    type="text"
                                    placeholder="Search by merchant name or total..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    className="w-full pl-12 pr-4 py-3 border-2 border-gray-200 rounded-xl bg-white text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                                />
                            </div>

                            {/* Filter */}
                            <select
                                value={filterProcessed}
                                onChange={(e) => setFilterProcessed(e.target.value)}
                                className="px-4 py-3 border-2 border-gray-200 rounded-xl bg-white text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                            >
                                <option value="all">All Receipts</option>
                                <option value="processed">Processed</option>
                                <option value="unprocessed">Unprocessed</option>
                            </select>

                            {/* Sort */}
                            <select
                                value={sortBy}
                                onChange={(e) => setSortBy(e.target.value)}
                                className="px-4 py-3 border-2 border-gray-200 rounded-xl bg-white text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                            >
                                <option value="date-desc">Newest First</option>
                                <option value="date-asc">Oldest First</option>
                                <option value="total-desc">Highest Total</option>
                                <option value="total-asc">Lowest Total</option>
                                <option value="merchant-asc">Merchant A-Z</option>
                                <option value="merchant-desc">Merchant Z-A</option>
                            </select>

                            {/* View Toggle */}
                            <div className="flex items-center gap-2 border-2 border-gray-200 rounded-xl p-1 bg-gray-50">
                                <button
                                    onClick={() => setViewMode('card')}
                                    className={`px-4 py-2 rounded-lg transition-all ${viewMode === 'card'
                                            ? 'bg-blue-600 text-white shadow-md'
                                            : 'text-gray-600 hover:bg-gray-100'
                                        }`}
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                                    </svg>
                                </button>
                                <button
                                    onClick={() => setViewMode('list')}
                                    className={`px-4 py-2 rounded-lg transition-all ${viewMode === 'list'
                                            ? 'bg-blue-600 text-white shadow-md'
                                            : 'text-gray-600 hover:bg-gray-100'
                                        }`}
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
                                    </svg>
                                </button>
                            </div>
                        </div>

                        <div className="flex items-center justify-between">
                            <p className="text-sm font-semibold text-gray-700">
                                Showing <span className="font-bold text-gray-900">{sortedReceipts.length}</span> of <span className="font-bold text-gray-900">{receipts.length}</span> receipts
                            </p>
                        </div>
                    </div>

                    {/* Receipts Display */}
                    {loading ? (
                        <div className="text-center py-16">
                            <div className="inline-block w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
                            <p className="text-gray-600 font-medium">Loading receipts...</p>
                        </div>
                    ) : sortedReceipts.length === 0 ? (
                        <div className="text-center py-16">
                            <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6 border-2 border-gray-200">
                                <svg className="w-10 h-10 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                            </div>
                            <h3 className="text-xl font-bold text-gray-900 mb-2">No receipts found</h3>
                            <p className="text-gray-700 mb-6 font-medium">Get started by uploading your first receipt</p>
                            <Button onClick={() => navigate('/upload')} size="lg">
                                Upload Your First Receipt
                            </Button>
                        </div>
                    ) : viewMode === 'card' ? (
                        // Card View
                        <div className="space-y-4">
                            {/* Select All Checkbox */}
                            <div className="flex items-center space-x-3 pb-4 border-b border-gray-200">
                                <input
                                    type="checkbox"
                                    checked={selectedReceipts.size === sortedReceipts.length && sortedReceipts.length > 0}
                                    onChange={handleSelectAll}
                                    className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                                />
                                <label className="text-sm font-semibold text-gray-700">Select All</label>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                {sortedReceipts.map(receipt => (
                                    <div
                                        key={receipt.id}
                                        className={`group bg-white border-2 rounded-xl p-6 hover:shadow-xl transition-all duration-300 cursor-pointer ${selectedReceipts.has(receipt.id)
                                                ? 'border-blue-500 bg-blue-50'
                                                : 'border-gray-200 hover:border-blue-300'
                                            }`}
                                        onClick={() => handleSelectReceipt(receipt.id)}
                                    >
                                        <div className="flex items-start justify-between mb-4">
                                            <div className="flex items-center space-x-3 flex-1">
                                                <input
                                                    type="checkbox"
                                                    checked={selectedReceipts.has(receipt.id)}
                                                    onChange={() => handleSelectReceipt(receipt.id)}
                                                    onClick={(e) => e.stopPropagation()}
                                                    className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                                                />
                                                <div className="flex-1">
                                                    <h3 className="text-lg font-bold text-gray-900 mb-2 line-clamp-1">
                                                        {receipt.merchant_name || 'Unknown Merchant'}
                                                    </h3>
                                                    {receipt.processed ? (
                                                        <span className="inline-flex items-center px-2.5 py-1 text-xs font-semibold bg-green-100 text-green-800 rounded-full border border-green-300">
                                                            <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                                            </svg>
                                                            Processed
                                                        </span>
                                                    ) : (
                                                        <span className="inline-flex items-center px-2.5 py-1 text-xs font-semibold bg-yellow-100 text-yellow-800 rounded-full border border-yellow-300">
                                                            <svg className="w-3 h-3 mr-1 animate-spin" fill="none" viewBox="0 0 24 24">
                                                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                            </svg>
                                                            Pending
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>

                                        <div className="space-y-3 mb-4">
                                            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-100">
                                                <div className="flex items-center space-x-2">
                                                    <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                    </svg>
                                                    <span className="text-sm font-semibold text-gray-700">Total</span>
                                                </div>
                                                <span className="text-xl font-bold text-blue-700">
                                                    ${receipt.total || 'N/A'}
                                                </span>
                                            </div>

                                            <div className="flex items-center space-x-4 text-sm">
                                                <div className="flex items-center space-x-2 text-gray-600">
                                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                                                        <path strokeLinecap="round" strokeLinejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                                    </svg>
                                                    <span>{formatDate(receipt.date || receipt.created_at)}</span>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="flex gap-2">
                                            <Button
                                                className="flex-1"
                                                size="sm"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    navigate(`/receipt/${receipt.id}`);
                                                }}
                                            >
                                                View
                                            </Button>
                                            <Button
                                                size="sm"
                                                color="danger"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleDeleteClick(receipt.id);
                                                }}
                                            >
                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                </svg>
                                            </Button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        // List View
                        <div className="space-y-3">
                            {/* Select All Checkbox */}
                            <div className="flex items-center space-x-3 pb-4 border-b border-gray-200">
                                <input
                                    type="checkbox"
                                    checked={selectedReceipts.size === sortedReceipts.length && sortedReceipts.length > 0}
                                    onChange={handleSelectAll}
                                    className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                                />
                                <label className="text-sm font-semibold text-gray-700">Select All</label>
                            </div>

                            {/* Table Header */}
                            <div className="hidden md:grid md:grid-cols-12 gap-4 px-6 py-3 bg-gray-50 rounded-lg border border-gray-200 font-semibold text-sm text-gray-700">
                                <div className="col-span-1"></div>
                                <div className="col-span-3">Merchant</div>
                                <div className="col-span-2">Date</div>
                                <div className="col-span-2">Total</div>
                                <div className="col-span-2">Status</div>
                                <div className="col-span-2">Actions</div>
                            </div>

                            {sortedReceipts.map(receipt => (
                                <div
                                    key={receipt.id}
                                    className={`group grid grid-cols-1 md:grid-cols-12 gap-4 items-center p-4 border-2 rounded-xl transition-all duration-300 ${selectedReceipts.has(receipt.id)
                                            ? 'bg-blue-50 border-blue-500'
                                            : 'bg-white border-gray-200 hover:shadow-lg hover:border-blue-300'
                                        }`}
                                >
                                    <div className="col-span-1">
                                        <input
                                            type="checkbox"
                                            checked={selectedReceipts.has(receipt.id)}
                                            onChange={() => handleSelectReceipt(receipt.id)}
                                            className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                                        />
                                    </div>

                                    <div className="col-span-1 md:col-span-3">
                                        <div className="flex items-center space-x-3">
                                            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                                                <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                                                </svg>
                                            </div>
                                            <div>
                                                <h3 className="font-bold text-gray-900 text-lg">
                                                    {receipt.merchant_name || 'Unknown Merchant'}
                                                </h3>
                                                <p className="text-xs text-gray-500 md:hidden">{formatDateTime(receipt.date || receipt.created_at)}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="col-span-1 md:col-span-2">
                                        <div className="flex items-center space-x-2 text-gray-700">
                                            <svg className="w-4 h-4 text-gray-500 hidden md:block" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                            </svg>
                                            <span className="font-medium">{formatDate(receipt.date || receipt.created_at)}</span>
                                        </div>
                                    </div>

                                    <div className="col-span-1 md:col-span-2">
                                        <div className="flex items-center space-x-2">
                                            <svg className="w-4 h-4 text-gray-500 hidden md:block" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                            </svg>
                                            <span className="font-bold text-lg text-gray-900">${receipt.total || 'N/A'}</span>
                                        </div>
                                    </div>

                                    <div className="col-span-1 md:col-span-2">
                                        {receipt.processed ? (
                                            <span className="inline-flex items-center px-3 py-1 text-xs font-semibold bg-green-100 text-green-800 rounded-full border border-green-300">
                                                <svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                                </svg>
                                                Processed
                                            </span>
                                        ) : (
                                            <span className="inline-flex items-center px-3 py-1 text-xs font-semibold bg-yellow-100 text-yellow-800 rounded-full border border-yellow-300">
                                                <svg className="w-3 h-3 mr-1 animate-spin" fill="none" viewBox="0 0 24 24">
                                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                </svg>
                                                Pending
                                            </span>
                                        )}
                                    </div>

                                    <div className="col-span-1 md:col-span-2">
                                        <div className="flex gap-2">
                                            <Button
                                                size="sm"
                                                onClick={() => navigate(`/receipt/${receipt.id}`)}
                                                className="flex-1"
                                            >
                                                View
                                            </Button>
                                            <Button
                                                size="sm"
                                                color="danger"
                                                onClick={() => handleDeleteClick(receipt.id)}
                                            >
                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                </svg>
                                            </Button>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </Card>
            </div>

            {/* Delete Confirmation Modal */}
            {showDeleteConfirm && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-2xl p-8 max-w-md w-full border-2 border-gray-200 shadow-2xl">
                        <div className="text-center mb-6">
                            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                            </div>
                            <h3 className="text-2xl font-bold text-gray-900 mb-2">
                                Confirm Delete
                            </h3>
                            <p className="text-gray-700">
                                {deleteTarget === null
                                    ? `Are you sure you want to delete ${selectedReceipts.size} receipt(s)? This action cannot be undone.`
                                    : 'Are you sure you want to delete this receipt? This action cannot be undone.'
                                }
                            </p>
                        </div>
                        <div className="flex gap-3">
                            <Button
                                onClick={() => {
                                    setShowDeleteConfirm(false);
                                    setDeleteTarget(null);
                                }}
                                color="outline"
                                className="flex-1"
                                disabled={deleting}
                            >
                                Cancel
                            </Button>
                            <Button
                                onClick={handleDeleteConfirm}
                                color="danger"
                                className="flex-1"
                                disabled={deleting}
                            >
                                {deleting ? 'Deleting...' : 'Delete'}
                            </Button>
                        </div>
                    </div>
                </div>
            )}

            {/* Success Modal */}
            {showSuccessModal && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-2xl p-8 max-w-md w-full border-2 border-green-200 shadow-2xl">
                        <div className="text-center mb-6">
                            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                            <h3 className="text-2xl font-bold text-gray-900 mb-2">
                                Success
                            </h3>
                            <p className="text-gray-700">
                                {modalMessage}
                            </p>
                        </div>
                        <div className="flex justify-center">
                            <Button
                                onClick={() => setShowSuccessModal(false)}
                                className="min-w-[120px]"
                            >
                                OK
                            </Button>
                        </div>
                    </div>
                </div>
            )}

            {/* Error Modal */}
            {showErrorModal && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                    <div className="bg-white rounded-2xl p-8 max-w-md w-full border-2 border-red-200 shadow-2xl">
                        <div className="text-center mb-6">
                            <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                            <h3 className="text-2xl font-bold text-gray-900 mb-2">
                                Error
                            </h3>
                            <p className="text-gray-700">
                                {modalMessage}
                            </p>
                        </div>
                        <div className="flex justify-center">
                            <Button
                                onClick={() => setShowErrorModal(false)}
                                color="danger"
                                className="min-w-[120px]"
                            >
                                OK
                            </Button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default ReceiptHistory;
