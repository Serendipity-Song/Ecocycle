package com.ecobyte.ecocycle.application;

import com.ecobyte.ecocycle.domain.quiz.DailyQuiz;
import com.ecobyte.ecocycle.domain.quiz.DailyQuizRepository;
import com.ecobyte.ecocycle.domain.quiz.Quiz;
import com.ecobyte.ecocycle.domain.quiz.QuizRepository;
import com.ecobyte.ecocycle.domain.user.User;
import com.ecobyte.ecocycle.domain.user.UserRepository;
import com.ecobyte.ecocycle.dto.request.DailyQuizAnswerRequest;
import com.ecobyte.ecocycle.dto.request.QuizRequest;
import com.ecobyte.ecocycle.dto.response.DailyQuizResponse;
import com.ecobyte.ecocycle.dto.response.QuizResponse;
import com.ecobyte.ecocycle.exception.AlreadyExistedDailyQuizException;
import com.ecobyte.ecocycle.exception.DailyQuizNotFoundException;
import com.ecobyte.ecocycle.exception.DailyQuizOwnedException;
import com.ecobyte.ecocycle.exception.NoQuizException;
import com.ecobyte.ecocycle.exception.UserNotFoundException;
import java.time.LocalDate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional(readOnly = true)
public class QuizService {

    private final QuizRepository quizRepository;
    private final UserRepository userRepository;
    private final DailyQuizRepository dailyQuizRepository;

    public QuizService(final QuizRepository quizRepository,
                       final UserRepository userRepository,
                       final DailyQuizRepository dailyQuizRepository) {
        this.quizRepository = quizRepository;
        this.userRepository = userRepository;
        this.dailyQuizRepository = dailyQuizRepository;
    }

    @Transactional
    public QuizResponse add(final QuizRequest quizRequest) {
        final Quiz quiz = new Quiz(quizRequest.getContent(), quizRequest.getAnswer(), quizRequest.getTip());
        final Quiz savedQuiz = quizRepository.save(quiz);
        return QuizResponse.from(savedQuiz);
    }

    @Transactional
    public DailyQuizResponse giveDailyQuiz(final Long loginId) {
        final LocalDate currentDate = LocalDate.now();
        checkDailyQuizAlreadyDid(loginId, currentDate);

        final DailyQuiz dailyQuiz = makeDailyQuiz(loginId, currentDate);
        return new DailyQuizResponse(dailyQuiz.getId(), QuizResponse.from(dailyQuiz.getQuiz()));

    }

    @Transactional
    public void updateDailyAnswer(final Long loginId, final Long dailyQuizId,
                                  final DailyQuizAnswerRequest dailyQuizAnswerRequest) {
        final DailyQuiz dailyQuiz = dailyQuizRepository.findById(dailyQuizId)
                .orElseThrow(DailyQuizNotFoundException::new);

        if (!dailyQuiz.isOwnedBy(loginId)) {
            throw new DailyQuizOwnedException();
        }

        dailyQuiz.updateAnswer(dailyQuizAnswerRequest.getIsRight());
    }

    private void checkDailyQuizAlreadyDid(final Long loginId, final LocalDate currentDate) {
        final boolean isAlreadyExistedDailyQuiz = dailyQuizRepository
                .existsByUserIdAndAttendanceDate(loginId, currentDate);

        if (isAlreadyExistedDailyQuiz) {
            throw new AlreadyExistedDailyQuizException();
        }
    }

    private DailyQuiz makeDailyQuiz(final Long userId, final LocalDate attendanceDate) {
        final Quiz quiz = quizRepository.findByRandomOne()
                .orElseThrow(NoQuizException::new);
        final User user = userRepository.findById(userId)
                .orElseThrow(UserNotFoundException::new);
        final DailyQuiz dailyQuiz = new DailyQuiz(user, quiz, attendanceDate);
        return dailyQuizRepository.save(dailyQuiz);
    }
}
