package com.ecobyte.ecocycle.domain.user;

import jakarta.persistence.Embeddable;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Embeddable
@NoArgsConstructor
@Getter
public class Stamp {

    static final Integer LIMIT_MAX = 40;
    private static final String CONSTRUCTOR_OVER_LIMIT_ERROR_MESSAGE = "스탬프의 초기값은 " + LIMIT_MAX + "을 넘을 수 없습니다.";

    private Integer stamps;

    public Stamp(final Integer stamps) {
        if (stamps >= LIMIT_MAX) {
            throw new IllegalArgumentException(CONSTRUCTOR_OVER_LIMIT_ERROR_MESSAGE);
        }

        this.stamps = stamps;
    }

    public boolean isFull(final Integer newStamps) {
        return stamps + newStamps >= LIMIT_MAX;
    }

    // TODO: stamps 추가하는 규약도 다 Stamp 클래스로 옮기는 방법 생각
    public void add(final Integer newStamps) {
        stamps = (stamps + newStamps) % LIMIT_MAX;
    }
}
