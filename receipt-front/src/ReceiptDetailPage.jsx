import { useNavigate,useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';

function ReceiptDetailPage() {
    const navigate = useNavigate();
    const { id } = useParams();//gets id from receipt/:id

    const [receipt,setReceipt] = useState(null);
    const [loading,setLoading] =useState(true);
    const [error,setError] = useState(null)

    useEffect(()=>{
        const loadReceipt = async()=>{
            try {
                const res = await fetch(`http://127.0.0.1:8000/api/receipts/${id}/`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);

                const data = await res.json()
                setReceipt(data);  
            } catch (e) {
                setError(String(e))
            } finally {
                setLoading(false);
            }
        };

        loadReceipt();
    }, [id]);

    if (loading){
        return (
            <div>
                <h1>Receipt Details</h1>
                <p>Loading receipt...</p>
                <button onClick={() => navigate('/')}>Back to Home</button>
            </div>
        );
    }
    if (error) {
        return (
            <div>
                <h1>Receipt Details</h1>
                <p style={{color:'red'}}>Error: {error}</p>
                <button onClick={()=>navigate('/receipts')}>Back to History</button>
            </div>
        
        )
    }
    if (!receipt) {
        return (
            <div>
                <h1>Receipt Details</h1>
                <p>Receipt not found</p>
                <button onClick={()=>navigate('/receipts')}>Back to History</button>
            </div>
        )
    }

    return (
        <div>
            <h1>Receipt Details</h1>


            <div style={{border: '1px solid #ccc', padding: '15px', margin: '10px 0' }}>
                <h2>Merchant Information</h2>
                <p><strong>Name: </strong>{receipt.merchant_name || 'N/A'}</p>
                <p><strong>Address:</strong>{receipt.merchant_address}</p>
                <p><strong>Phone:</strong> {receipt.merchant_phone || 'N/A'}</p>
                <p><strong>Email:</strong> {receipt.merchant_email || 'N/A'}</p>
            </div>

            <div style={{ border: '1px solid #ccc', padding: '15px', margin: '10px 0' }}>
                <h2>Transaction Information</h2>
                <p><strong>Date:</strong> {receipt.date || 'N/A'}</p>
                <p><strong>Time:</strong> {receipt.time || 'N/A'}</p>
                <p><strong>Receipt Number:</strong> {receipt.receipt_number || 'N/A'}</p>
                <p><strong>Transaction ID:</strong> {receipt.transaction_id || 'N/A'}</p>
            </div>

              {/* Financial Info */}
            <div style={{ border: '1px solid #ccc', padding: '15px', margin: '10px 0' }}>
                <h2>Financial Details</h2>
                <p><strong>Subtotal:</strong> ${receipt.sub_total || 'N/A'}</p>
                <p><strong>Tax:</strong> ${receipt.tax || 'N/A'}</p>
                <p><strong>Tip:</strong> ${receipt.tip || 'N/A'}</p>
                <p><strong>Discount:</strong> ${receipt.discount || 'N/A'}</p>
                <p><strong>Total:</strong> ${receipt.total || 'N/A'}</p>
                <p><strong>Currency:</strong> {receipt.currency || 'N/A'}</p>
            </div>
            {/* Payment Info */}
            <div style={{ border: '1px solid #ccc', padding: '15px', margin: '10px 0' }}>
                <h2>Payment Information</h2>
                <p><strong>Payment Method:</strong> {receipt.payment_method || 'N/A'}</p>
                <p><strong>Card Type:</strong> {receipt.card_type || 'N/A'}</p>
                <p><strong>Card Last Four:</strong> {receipt.card_last_four || 'N/A'}</p>
            </div>

            {/* Items */}
            <div style={{ border: '1px solid #ccc', padding: '15px', margin: '10px 0' }}>
                <h2>Items</h2>
                {receipt.items && receipt.items.length > 0 ? (
                  receipt.items.map((item, index) => (
                    <div key={index} style={{ margin: '5px 0' }}>
                        <p><strong>Item:</strong> {item.name || 'Unknown'}</p>
                        <p><strong>Price:</strong> ${item.price || 'N/A'}</p>
                        <p><strong>Quantity:</strong> {item.quantity || 'N/A'}</p>
                    </div>
                ))
               ) : (
                <p>No items extracted</p>
               )}
            </div>
            {/* Navigation */}
            <div style={{ margin: '20px 0' }}>
                <button onClick={() => navigate('/receipts')}>Back to History</button>
                <button onClick={() => navigate('/')} style={{ marginLeft: '10px' }}>Home</button>
            </div>
        </div>
    );
}

export default ReceiptDetailPage;
