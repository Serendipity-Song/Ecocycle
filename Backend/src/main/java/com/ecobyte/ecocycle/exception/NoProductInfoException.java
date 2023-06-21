package com.ecobyte.ecocycle.exception;

import lombok.Getter;

@Getter
public class NoProductInfoException extends RuntimeException {
    private final String message;

    public NoProductInfoException() {
        this.message = "정보가 없습니다.";
    }
}
