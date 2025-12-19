import { useNavigate } from 'react-router-dom';
import { useEffect,useState } from 'react';
import { Page, Card, Button } from './components';


function Home(){
    const navigate=useNavigate();
    
    return(
        <Page>
            <Card>
            <h1 className="text-2xl font-bold text-center mb-6">Home Page</h1>
            <div className="space-y-4">
                <Button onClick={()=>navigate('/upload')} className="w-full">Process a Receipt</Button>
                <Button onClick={()=>navigate('/receipts')} color='success' className="w-full">
                    See Processed Receipt History</Button>
            </div>
            </Card>
        </Page>
        
    );
}

export default Home;
