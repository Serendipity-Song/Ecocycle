import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Login from './pages/login';
import Main from './pages/main';
import Admin from './pages/admin';
import Report from './pages/report';
import Mypage from './pages/mypage';
import AddProduct from './pages/addProduct';
import AddQuiz from './pages/addQuiz';
import { Navigate } from "react-router-dom";

const isLogin = false;

function App() {
    return (
        <div className="App">
            <BrowserRouter>
                <Routes>
                    <Route path={"/"} element={isLogin ? <Navigate to="/main" /> : <Navigate to="/login" />} ></Route>
                    <Route path={"/login"} element={<Login />}></Route>
                    <Route path={"/main"} element={<Main />}></Route>
                    <Route path={"/report"} element={<Report />}></Route>
                    <Route path={"/mypage"} element={<Mypage />}></Route>
                    <Route path={"/admin"} element={<Admin />}></Route>
                    <Route path={"/addProduct"} element={<AddProduct />}></Route>
                    <Route path={"/addQuiz"} element={<AddQuiz />}></Route>
                </Routes>
            </BrowserRouter >
        </div >
    );
}

export default App;