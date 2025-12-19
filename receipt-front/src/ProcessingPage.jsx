import { useNavigate,useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';


function useQuery(){
    const { search }= useLocation();
    return new URLSearchParams(search)
}

function ProcessingPage() {
    const navigate = useNavigate();
    const query = useQuery();
    const id=query.get('id');

    const [status,setStatus]=useState('idle')
    const [error,setError]=useState(null)

    useEffect(()=>{
        const runProcessing = async ()=>{
            if (!id) {
                setError('Missing Receipt ID')
                setStatus('error');
                return;
            }
            setStatus('processing')
            setError(null)

            try {
                const res = await fetch(`http://127.0.0.1:8000/api/receipts/${id}/process/`,{
                    method: 'POST',
                });

                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data=await res.json()
                console.log('Processing Result:',data)

                setStatus('Completed')
            } catch (e) {
                setError(String(e));
                setStatus('error');
            }
        };
        runProcessing();
    }, [id]);
    
    return (
        <div>
            <h1>Processing Receipt</h1>
            {status === 'processing' && (
                <div>
                    <p>Processing Receipt...</p>
                    <p>IN Progress</p>
                </div>
                    
            )}
            {status === 'Completed' &&(
                <div>
                    <p>Processing Complete</p>
                    <p>Data has been extracted</p>
                    <button onClick={()=>navigate(`/receipt/${id}`)}>View Details</button>
                </div>
            )}
            {status === 'error' && (
                <div>
                    <p style={{color: 'red'}}>Error: {error}</p>
                    <button onClick={()=>navigate('/upload')}>Try Again</button>
                </div>
            )}
            
            <button onClick={() => navigate('/')}>Back to Home</button>
        </div>
    );
}

export default ProcessingPage;
