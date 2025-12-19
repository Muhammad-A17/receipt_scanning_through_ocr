import { useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';

function ReceiptHistory() {
    const navigate = useNavigate();
    const [receipts,setReceipts] =useState([]);
    const [loading,setLoading]=useState(true);
    useEffect(() => {
        const load = async () => {
            try {
                const res=await fetch('http://127.0.0.1:8000/api/receipts/');
                if (!res.ok)throw new Error(`HTTP ${res.status}`);
                const raw = await res.json();
                const items= Array.isArray(raw) ? raw : (raw.results ?? []);
                setReceipts(items);
            } catch (err) {
                console.error('Error fetching receipts:',err);
            } finally {
                setLoading(false);
            }
        };
        load();
    }, []);
    
    return (
        <div>
            <h1>Receipt History</h1>
            <p>All processed receipts will be shown here</p>
            <button onClick={() => navigate('/')}>Back to Home</button>
            <h2>All Receipts</h2>
                {loading ? (
                    <p>Loading receipts...</p>
                ) : (
                    <div>
                        {receipts.length === 0 ? (
                            <p>No receipts found</p>
                        ) : (
                        receipts.map(receipt=>(
                        <div key={receipt.id} style={{
                            border: '2px solid #646cff',
                            borderRadius: '8px',
                            padding: '15px',
                            margin: '10px 0',
                            backgroundColor: '#1a1a1a',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center'
                        }}>
                            <div>
                                <h3 style={{margin: '0 0 5px 0', color: 'white'}}>
                                    {receipt.merchant_name || 'Unknown Merchant'}
                                </h3>
                                <p style={{margin: '0 0 5px 0', color: 'white'}}>
                                    Total: ${receipt.total || 'N/A'}
                                </p>
                                <p style={{margin: '0', color: 'white'}}>
                                    Date: {receipt.created_at}
                                </p>
                            </div>
                            <button onClick={() => navigate(`/receipt/${receipt.id}`)}>
                                View Details
                            </button>
                        </div>
                    ))     
                )}
            </div>
            )}
        </div>
    );
}

export default ReceiptHistory;
