import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './Home'
import UploadPage from './UploadPage' 
import ProcessingPage from './ProcessingPage'
import ReceiptDetailPage from './ReceiptDetailPage'
import ReceiptHistory from './ReceiptHistory'

function App(){
  return(
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path='/' element={<HomePage/>}/>
        <Route path='/Home' element={<HomePage/>}/>
        <Route path='/upload' element={<UploadPage/>}/>
        <Route path='/processing' element={<ProcessingPage/>}/>
        <Route path='/receipt/:id' element={<ReceiptDetailPage/>}/>
        <Route path='/receipts' element={<ReceiptHistory/>}/>
      </Routes>
    </BrowserRouter>
  );
}
export default App;
