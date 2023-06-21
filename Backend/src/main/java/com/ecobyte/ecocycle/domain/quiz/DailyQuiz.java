package com.ecobyte.ecocycle.domain.quiz;

import com.ecobyte.ecocycle.domain.user.User;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.time.LocalDate;
import lombok.AccessLevel;
import lombok.EqualsAndHashCode;
import lombok.EqualsAndHashCode.Include;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "daily_quiz")
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
public class DailyQuiz {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Include
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "quiz_id", nullable = false)
    private Quiz quiz;

    @Column(nullable = false)
    private Boolean isRight;

    @Column(nullable = false)
    private LocalDate attendanceDate;

    public DailyQuiz(final User user, final Quiz quiz, final LocalDate attendanceDate) {
        this.user = user;
        this.quiz = quiz;
        this.isRight = false;
        this.attendanceDate = attendanceDate;
    }

    public boolean isOwnedBy(final Long userId) {
        return user.isSameId(userId);
    }

    public void updateAnswer(final boolean isRight) {
        if (isRight) {
            user.addStamps(7);
        } else {
            user.addStamps(3);
        }

        this.isRight = isRight;
    }
}
