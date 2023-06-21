import '../styles/App.css';
import '../styles/reset.css';
import {useNavigate} from 'react-router-dom';
import {useEffect, useState} from 'react';
import makeMain from '../hooks/MakeMain';
import {getQuiz, getUserInfo, putQuizAnswer} from '../api/apis';
import DailyQuizModal from '../modals/dailyQuiz';
import Loading from '../Loading';

const Main = () => {
    const [loading, setLoading] = useState(true);
    const startLoading = () => setLoading(true);
    const endLoading = () => setLoading(false);

    const [photoPath, setPhotoPath] = useState("");


    const movePage = useNavigate();

    const [isQuizOpen, setIsQuizOpen] = useState(false);
    const [isAnswerRight, setIsAnswerRight] = useState(false);
    const [isExpOpen, setIsExpOpen] = useState(false);

    const openQuiz = () => setIsQuizOpen(true);

    const closeQuiz = () => setIsQuizOpen(false);

    const answerRight = () => setIsAnswerRight(true);
    const expOpen = () => setIsExpOpen(true);
    const expClose = () => setIsExpOpen(false);


    const [quizId, setQuizId] = useState("");
    const [content, setContent] = useState(false);
    const [answer, setAnswer] = useState(false);
    const [tip, setTip] = useState(false);


    const showTree = () => {

        const {stateOfTree} = makeMain();

        if (stateOfTree === 'seed') {
            setPhotoPath('assets/userSeed.PNG');
        } else if (stateOfTree === 'sprout') {
            setPhotoPath('assets/userSprout.PNG');
        } else if (stateOfTree === 'seedling') {
            setPhotoPath('assets/userSeedling.PNG');
        } else if (stateOfTree === 'tree') {
            setPhotoPath('assets/userTree.png');
        }
    }


    const showQuiz = async () => {

        console.log("showQuiz");

        if (localStorage.getItem('dailyQuiz') === 'false') {

            const {quizInfo} = await getQuiz();

            console.log(quizInfo);

            setQuizId(quizInfo.id);
            setContent(quizInfo.quiz.content);
            setAnswer(quizInfo.quiz.answer);
            setTip(quizInfo.quiz.tip);

            openQuiz();
        }
    };

    const userAnswer = (userAnswer) => {
        closeQuiz();
        if (answer === userAnswer) {
            answerRight();

            putQuizAnswer(quizId, true);
        } else {
            putQuizAnswer(quizId, false);
        }

        expOpen();
    }

    function showMenu() {
    }

    function goReport() {
        movePage('/report');
    }

    function goMypage() {
        movePage('/mypage');
    }

    useEffect(() => {
        const show = async () => {
            await getUserInfo();
            showTree();
            endLoading();
            showQuiz();
        }
        show();
    }, []);

    return (
        <div>
            {loading ? <Loading/> : null}
            <main id="main_page">
                <section>
                    <img id="logo_img" src="assets/logo.png" alt="EcoCycle logo"/> <br/>
                    <DailyQuizModal isOpen={isQuizOpen}>
                        <div className="quiz_modal">
                            <div className="dailyQuizModal_div">
                                <div>
                                    <p className="quiz_title"> EcoQuiz </p>
                                    <div id="quiz_info">
                                        <div className="quiz_content">Q. {content}</div>
                                        <button className="quiz_btn" onClick={() => userAnswer(true)}>O</button>
                                        <button className="quiz_btn" onClick={() => userAnswer(false)}> X</button>
                                    </div>
                                </div>
                                <br/>
                            </div>
                        </div>
                    </DailyQuizModal>

                    <div className="quiz_modal" style={{display: isExpOpen ? "block" : "none"}}>
                        <div className="dailyQuizModal_div">
                            <button className="modalClose_btn" onClick={expClose}>X</button>
                            <span className="quiz_span" style={{display: isAnswerRight ? "block" : "none"}}>
                                <p className="quiz_title"> EcoQuiz </p>
                                <p className="quiz_content"> 축하합니다! </p> <p
                                className="quiz_content"> 정답을 맞히셨습니다 </p> <br/>
                                <div className="quiz_tip">Tip. {tip}</div>
                            </span>

                            <span className="quiz_span" style={{display: isAnswerRight ? "none" : "block"}}>
                                <button className="modalClose_btn" onClick={expClose}>X</button>
                                <p className="quiz_title"> EcoQuiz </p>
                                <p className="quiz_content"> 아쉽게도 오답입니다ㅠㅠ </p> <br/>
                                <div className="quiz_tip">Tip. {tip}</div>
                            </span>
                        </div>
                    </div>

                    <img id="user_tree" src={`${photoPath}`} alt="User Tree"/>
                    <footer id="main_footer">
                        <input type="image" id="menu_btn" src="assets/menuBtn.png" alt="Menu Button"
                               onClick={showMenu}/>
                        <input type="image" id="camera_btn" src="assets/camerBtn.png" alt="Camera Button"
                               onClick={goReport}/>
                        <input type="image" id="mypage_btn" src="assets/stampBtn.png" alt="Mypage Button"
                               onClick={goMypage}/>
                    </footer>
                </section>
            </main>
        </div>
    );
}

export default Main;
