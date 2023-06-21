package com.ecobyte.ecocycle.domain.user;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class StampTest {

    @DisplayName("stamp가 추가될 때 limit를 넘는지 확인한다(넘는 경우)")
    @Test
    void isFull() {
        // given
        final Stamp stamp = new Stamp(0);

        // when, then
        assertThat(stamp.isFull(Stamp.LIMIT_MAX)).isTrue();
    }

    @DisplayName("stamp가 추가되면 limit를 넘는지 확인한다(넘지 않는 경우)")
    @Test
    void isNotFull() {
        // given
        final Stamp stamp = new Stamp(0);

        // when, then
        assertThat(stamp.isFull(Stamp.LIMIT_MAX - 1)).isFalse();
    }


}
