package com.ecobyte.ecocycle.application;

import com.ecobyte.ecocycle.domain.quiz.DailyQuizRepository;
import com.ecobyte.ecocycle.domain.user.User;
import com.ecobyte.ecocycle.domain.user.UserRepository;
import com.ecobyte.ecocycle.dto.response.MainPageResponse;
import com.ecobyte.ecocycle.exception.UserNotFoundException;
import java.time.LocalDate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional(readOnly = true)
public class UserService {

    private final UserRepository userRepository;
    private final DailyQuizRepository dailyQuizRepository;

    public UserService(final UserRepository userRepository,
                       final DailyQuizRepository dailyQuizRepository) {
        this.userRepository = userRepository;
        this.dailyQuizRepository = dailyQuizRepository;
    }

    public MainPageResponse findMyMainPage(final Long loginId) {
        final User loginUser = userRepository.findById(loginId)
                .orElseThrow(UserNotFoundException::new);

        final boolean alreadyDailyQuiz = dailyQuizRepository.existsByUserIdAndAttendanceDate(loginId, LocalDate.now());
        return new MainPageResponse(loginUser.getNickname(), loginUser.getStamp().getStamps(), alreadyDailyQuiz);
    }
}
