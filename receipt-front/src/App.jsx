import { BrowserRouter, Routes, Route, Link,Navigate } from 'react-router-dom';
import HomePage from './Home'
import UploadPage from './UploadPage' 
import ProcessingPage from './ProcessingPage'
import ReceiptDetailPage from './ReceiptDetailPage'
import ReceiptHistory from './ReceiptHistory'

function App(){
  return(
    <BrowserRouter>
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo/Brand */}
            <div className="flex-shrink-0">
              <Link to='/' className="text-white text-xl font-bold hover:text-blue-400 transition-colors">
                📄 Receipt Scanner
              </Link>
            </div>
            
            {/* Navigation Links */}
            <div className="hidden md:block">
              <div className="ml-10 flex items-baseline space-x-4">
                <Link 
                  to='/' 
                  className="text-gray-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors"
                >
                  Home
                </Link>
                <Link 
                  to='/upload' 
                  className="text-gray-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors"
                >
                  Upload
                </Link>
                <Link 
                  to='/receipts' 
                  className="text-gray-300 hover:bg-gray-700 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors"
                >
                  History
                </Link>
              </div>
            </div>
            
            {/* Mobile menu button */}
            <div className="md:hidden">
              <button className="text-gray-400 hover:text-white hover:bg-gray-700 p-2 rounded-md">
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </nav>
      <Routes>
        {/* Home Page */}
        <Route path='/' element={<HomePage/>}/>
        <Route path='/Home' element={<HomePage/>}/>
        
        {/*Upload Page*/}
        <Route path='/upload' element={<UploadPage/>}/>
        
        {/*ProcessingPage*/}
        <Route path='/processing' element={<ProcessingPage/>}/>
        
        {/*ReceiptDetailPage*/}
        <Route path='/receipt/:id' element={<ReceiptDetailPage/>}/>

        {/*ReceiptHistoryPage*/}
        <Route path='/receipts' element={<ReceiptHistory/>}/>
        
      </Routes>
    </BrowserRouter>

  );
}
export default App;


