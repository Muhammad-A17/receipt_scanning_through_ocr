/**
 * Centralized API Service
 * Handles all network requests to the backend
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

const api = {
    /**
     * Fetch all receipts
     */
    getReceipts: async () => {
        const res = await fetch(`${BASE_URL}/receipts/`);
        if (!res.ok) throw new Error('Failed to fetch receipts');
        return res.json();
    },

    /**
     * Get single receipt details
     */
    getReceipt: async (id) => {
        const res = await fetch(`${BASE_URL}/receipts/${id}/`);
        if (!res.ok) throw new Error(`Failed to fetch receipt ${id}`);
        return res.json();
    },

    /**
     * Upload an image
     */
    uploadReceipt: async (file) => {
        const formData = new FormData();
        formData.append('image', file);
        const res = await fetch(`${BASE_URL}/receipts/upload/`, {
            method: 'POST',
            body: formData,
        });
        if (!res.ok) throw new Error('Upload failed');
        return res.json();
    },

    /**
     * Process a receipt with OCR
     */
    processReceipt: async (id) => {
        const res = await fetch(`${BASE_URL}/receipts/${id}/process/`, {
            method: 'POST',
        });
        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            throw new Error(errorData.error || errorData.message || 'Processing failed');
        }
        return res.json();
    },

    /**
     * Update receipt data
     */
    updateReceipt: async (id, data) => {
        const res = await fetch(`${BASE_URL}/receipts/${id}/edit/`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error('Update failed');
        return res.json();
    },

    /**
     * Delete a receipt
     */
    deleteReceipt: async (id) => {
        const res = await fetch(`${BASE_URL}/receipts/${id}/`, {
            method: 'DELETE',
        });
        if (!res.ok) throw new Error('Delete failed');
        return res.json();
    },

    /**
     * Bulk delete receipts
     */
    bulkDelete: async (ids) => {
        const res = await fetch(`${BASE_URL}/receipts/bulk-delete/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids }),
        });
        if (!res.ok) throw new Error('Bulk delete failed');
        return res.json();
    }
};

export default api;
