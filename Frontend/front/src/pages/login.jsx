import useGoogle from '../hooks/UseGoogle';
import '../styles/App.css';
import '../styles/reset.css';

const CI = process.env.REACT_APP_CLIENT_ID;

const Login = () => {

    const { loginUrl } = useGoogle({
        clientId: CI,
    })

    return (
        <div>
            <main>
                <hr className="top_line" />
                <section>
                    <img id="logo_img" src="assets/logo.png" alt="EcoCycle logo" /> <br />
                    <button className="user_btn" id="login_btn" onClick={() => { window.location.href = loginUrl }}>Google로 로그인</button>
                </section>
                <footer id='login_footer'>
                    <hr className="bottom_line" />
                    <img id="copyright_img" src="assets/copyright.png" alt="Copyright by EcoByte" />
                </footer>
            </main>
        </div>
    );
}

export default Login;