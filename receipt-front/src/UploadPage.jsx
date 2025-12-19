import { useNavigate } from 'react-router-dom';
import { useState } from 'react';

function UploadPage() {
    const navigate = useNavigate();

    //UI state
    const [file,setFile]=useState(null);//holds the selectd image
    const [uploading,setUploading]=useState(false);
    const [error,setError]=useState(null);
    const [imagePreview,setImagePreview]=useState(null);

    const [receiptId,setReceiptId]=useState(null);

    const onFileChange = (e) => {
        const selectedFile = e.target.files?.[0] ?? null;
        setFile(selectedFile);
        setError(null);
        
        // Create preview
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = (e) => {
                setImagePreview(e.target.result);
            };
            reader.readAsDataURL(selectedFile);
        } else {
            setImagePreview(null);
        }
    };

    const onUpload =async()=>{
        if (!file) {setError('Choose an image first');return;}

        setUploading(true);
        setError(null);
        try {
            const form = new FormData();
            form.append('image',file);

            const res =await fetch('http://127.0.0.1:8000/api/receipts/upload/',{
                method: 'POST',
                body: form,
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const data=await res.json();
            setReceiptId(data.id);
        } catch (e) {
            setError(String(e));
        } finally {
            setUploading(false);
        }
        
    };
    const goToProcessing =()=>{
        if (receiptId) navigate(`/processing?id=${receiptId}`);
    };
    
    return (
        <div>
            <h1>Upload Receipt</h1>
            <p>Upload your receipt image here</p>
            <input type='file' accept='image/*' onChange={onFileChange}/>
            
            {/* Image Preview */}
            {imagePreview && (
                <div>
                    <h3>Preview:</h3>
                    <img src={imagePreview} alt="Receipt preview" style={{maxWidth: '300px', height: 'auto'}} />
                </div>
            )}
            
            <button disabled={uploading} onClick={onUpload}>{uploading ? 'Uploading...' : 'Upload'}</button>
            {error && <p style={{ color: 'red' }}>{error}</p>}
            

            {receiptId && (
                <div>
                    <p>Upload Successful. Receipt ID: {receiptId}</p>
                    <button onClick={goToProcessing}>Process Receipt</button>
                </div>
            )}
            <button onClick={() => navigate('/')}>Back to Home</button>
        </div>
    );
}

export default UploadPage;
