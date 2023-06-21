import '../styles/App.css';
import '../styles/reset.css';
import { useNavigate } from 'react-router-dom';
import React, { useRef, useState } from "react";
import imageHandler from '../hooks/ImageHandler';
import { getProductInfo } from '../api/apis';
import ProductInfoModal from '../modals/productInfo';
import GoReportModal from '../modals/goReport';
import Loading from '../Loading';


const Report = () => {

    const [loading, setLoading] = useState(false);
    const startLoading = () => setLoading(true);
    const endLoading = () => setLoading(false);

    const movePage = useNavigate();

    function goMain() {
        movePage('/main');
    }

    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isGoReportModalOpen, setIsGoReportModalOpen] = useState(false);

    const [name, setName] = useState(false);
    const [rI, setrI] = useState(false);
    const [tip, setTip] = useState(false);

    const [name2, setName2] = useState(false);
    const [rI2, setrI2] = useState(false);
    const [tip2, setTip2] = useState(false);

    const [isSecond, setIsSecond] = useState(false);
    const [showSecond, setShowSecond] = useState(false);

    const showSecondProduct = () => {
        setShowSecond(true);
    }

    const showFirstProduct = () => {
        setShowSecond(false);
    }

    const openModal = () => setIsModalOpen(true);
    const closeModal = () => setIsModalOpen(false);

    const openGoReportModal = () => setIsGoReportModalOpen(true);

    const [imgFile, setImgFile] = useState("");
    const imgRef = useRef();


    const saveImgFile = async () => {

        const file = imgRef.current.files[0];
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            setImgFile(reader.result);
        };

        startLoading();

        const presignedUrl = await imageHandler(file);
        localStorage.setItem('presignedUrl', presignedUrl);
        const { info } = await getProductInfo(presignedUrl);


        if (info.products.length === 0) {
            openGoReportModal();
            endLoading();
        }
        else if (info.products.length > 0) {
            setName(info.products[0].name);
            setrI(info.products[0].recyclingInfo);
            setTip(info.products[0].tip);

            if (info.products.length === 2) {
                setName2(info.products[1].name);
                setrI2(info.products[1].recyclingInfo);
                setTip2(info.products[1].tip);

                setIsSecond(true);
            }
            endLoading();
            openModal();
        }
    };


    return (
        <div>
            {loading ? <Loading /> : null}
            <main>
                <div id="wrapper">
                    <img id="logo2_img" src="assets/logo2.png" alt="EcoCycle logo" onClick={goMain} /> <br />
                    <hr className="top_line" />
                    <section>
                        <span id="report_span">
                            <img id="upload_img" src={imgFile ? imgFile : "assets/report.png"} alt="Upload" />
                        </span>
                        <form>
                            <label className="uploadImage_label" htmlFor="uploadImage_input"> 사진 첨부하기</label>
                            <input type="file" id="uploadImage_input" accept="image/*" style={{ display: "none" }}
                                ref={imgRef} onChange={saveImgFile} />
                        </form>
                    </section>
                    <ProductInfoModal isOpen={isModalOpen} closeModal={closeModal}>
                        <div>
                            <div id="productInfoModal_span">
                                <button className="modalClose_btn" onClick={closeModal}>X</button>
                                <span className="modal_text" style={{ display: showSecond ? "none" : "block" }}>
                                    <div>
                                        <div className='productName_div'>
                                            <span id="title_text"  >이것은 </span>
                                            <span id="name_text">{name}</span>
                                            <span id="title_text" >입니다</span>
                                        </div> <br />
                                        <div className="productInfo_text" >{rI}</div> <br />
                                        <div className="tip_text" >Tip. {tip}</div>
                                    </div>
                                    <button className="result_btn" style={{ display: isSecond ? "block" : "none" }} onClick={showSecondProduct}>next</button>
                                </span>

                                <span className="modal_text" style={{ display: showSecond ? "block" : "none" }}>
                                    <div>
                                        <div className='productName_div'>
                                            <span id="title_text"  >이것은 </span>
                                            <span id="name_text">{name2}</span>
                                            <span id="title_text" >입니다</span>
                                        </div> <br />
                                        <div className="productInfo_text" >{rI2}</div> <br />
                                        <div className="tip_text" >Tip. {tip2}</div>
                                    </div>
                                    <button className="result_btn" onClick={showFirstProduct}>prev</button>
                                </span>
                            </div>
                        </div>
                    </ProductInfoModal>

                    <GoReportModal isOpen={isGoReportModalOpen}></GoReportModal>
                </div>

                <footer id="report_footer">
                    <hr className="bottom_line" />
                    <img id="copyright2_img" src="assets/copyright2.png" alt="Copyright by EcoByte" />
                </footer>
            </main>
        </div>
    );
}

export default Report;
