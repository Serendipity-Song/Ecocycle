package com.ecobyte.ecocycle.domain.user;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class UserTest {

    @DisplayName("사용자가 관리자인지 확인한다")
    @Test
    void isAdmin() {
        // given & when
        final User user = new User("azpi", "azpi", "azpi@email.com", Role.ADMIN);

        // then
        assertThat(user.isAdmin()).isTrue();
    }

    @DisplayName("사용자가 관리자가 아닌 경우인지 확인한다")
    @Test
    void isNotAdmin() {
        // given & when
        final User user = new User("azpi", "azpi", "azpi@email.com", Role.USER);

        // then
        assertThat(user.isAdmin()).isFalse();
    }


    @DisplayName("사용자의 Stamp를 증가시킨다(tree 생성)")
    @Test
    void addStamp_makeTree() {
        // given
        final User user = new User("azpi", "azpi", "azpi@email.com", Role.USER);

        // when
        user.addStamps(Stamp.LIMIT_MAX);

        // then
        assertThat(user.getTree()).isEqualTo(1);
    }
}
