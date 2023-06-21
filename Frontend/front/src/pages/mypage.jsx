import '../styles/App.css';
import '../styles/reset.css';
import { useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { getUserInfo } from '../api/apis';
import Loading from '../Loading';
import makeMain from '../hooks/MakeMain';

const Mypage = () => {

    const [loading, setLoading] = useState(true);
    const startLoading = () => setLoading(true);
    const endLoading = () => setLoading(false);

    const [stamp, setStamp] = useState("");

    const [photoPath, setPhotoPath] = useState("");
    const [photoPath2, setPhotoPath2] = useState("");

    const movePage = useNavigate();
    function goMain() {
        movePage('/main');
    }

    const showGrade = () => {
        const {stateOfTree} = makeMain();

        if (stateOfTree === 'seed') {
            setPhotoPath('assets/gradeSeed.PNG');
        } else if (stateOfTree === 'sprout') {
            setPhotoPath('assets/gradeSprout.PNG');
        } else if (stateOfTree === 'seedling') {
            setPhotoPath('assets/gradeSeedling.PNG');
        } else if (stateOfTree === 'tree') {
            setPhotoPath('assets/gradeTree.PNG');
        }
    }

    const showStamp = () => {
        const userStamp = localStorage.getItem('stamps');
        const remainder = userStamp % 10;

        let stampNum;

        switch(remainder) {
            case 0:
                stampNum = 0;
                break;
            case 1:
                stampNum = 1;
                break;
            case 2:
                stampNum = 2;
                break;
            case 3:
                stampNum = 3;
                break;
            case 4:
                stampNum = 4;
                break;
            case 5:
                stampNum = 5;
                break;
            case 6:
                stampNum = 6;
                break;
            case 7:
                stampNum = 7;
                break;
            case 8:
                stampNum = 8;
                break;
            case 9:
                stampNum = 9;
                break;
            default: 
        }
        setPhotoPath2(`assets/stamp${stampNum}.PNG`);

    }

    useEffect(() => {
        const saveUserInfo = async () => {
            await getUserInfo();
            setStamp(localStorage.getItem('stamps'));
            showGrade();
            showStamp();
        }
        saveUserInfo();
        endLoading();
    }, []);

    return (
        <div>
            {loading ? <Loading /> : null}
            <main id="main_page">
                <section id='mypage_section' >
                    <img id="mypage_logo" src="assets/logo.png" alt="EcoCycle logo" onClick={goMain} /> <br />
                    <img src="assets/myGrade.png" alt="나의 등급표시" />
                    <img id='grade_img' src={`${photoPath}`} alt="" />
                    <img id='stamp' src={`${photoPath2}`} alt="" />

                </section>
                <p className='text' id='total_stamp' >Total Stamp: {stamp} </p>
                <img id='arrow_img' src="assets/arrow.png" alt="뒤로 가기" onClick={goMain}  />
            </main>
        </div>
    );
}

export default Mypage;