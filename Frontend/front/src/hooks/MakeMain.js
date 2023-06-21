
const makeMain = () => {

    const getStateOfTree = () => {

        const userStamp = localStorage.getItem('stamps');

        if (userStamp >= 0 && userStamp < 10) {
            return 'seed';
        }
        else if (userStamp < 20) {
            return 'sprout';
        }
        else if (userStamp < 30) {
            return 'seedling';
        }
        else if (userStamp < 40) {
            return 'tree';
        }

    }

    const stateOfTree = getStateOfTree();

    return { stateOfTree };
}

export default makeMain;