package com.ecobyte.ecocycle.presentation;

import com.ecobyte.ecocycle.dto.response.ErrorResponse;
import com.ecobyte.ecocycle.exception.AlreadyExistedDailyQuizException;
import com.ecobyte.ecocycle.exception.DailyQuizOwnedException;
import com.ecobyte.ecocycle.exception.InvalidImageUrlException;
import com.ecobyte.ecocycle.exception.NoQuizException;
import com.ecobyte.ecocycle.exception.UserNotFoundException;
import com.ecobyte.ecocycle.exception.auth.AdminAuthorizationException;
import com.ecobyte.ecocycle.exception.auth.TokenException;
import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class ControllerAdvice {

    @ExceptionHandler(TokenException.class)
    public ResponseEntity<ErrorResponse> handleExpiredTokenException(final TokenException exception) {
        return ResponseEntity.status(exception.getHttpStatus())
                .body(new ErrorResponse(exception.getMessage()));
    }

    @ExceptionHandler(UserNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleExpiredTokenException(final UserNotFoundException exception) {
        return ResponseEntity.status(exception.getHttpStatus())
                .body(new ErrorResponse(exception.getMessage()));
    }

    @ExceptionHandler(AdminAuthorizationException.class)
    public ResponseEntity<ErrorResponse> handleAdminException(final AdminAuthorizationException exception) {
        return ResponseEntity.status(exception.getHttpStatus())
                .body(new ErrorResponse(exception.getMessage()));
    }

    @ExceptionHandler(InvalidImageUrlException.class)
    public ResponseEntity<ErrorResponse> handleInvalidImageUrlException(final InvalidImageUrlException exception) {
        return ResponseEntity.status(exception.getHttpStatus())
                .body(new ErrorResponse(exception.getMessage()));
    }

    @ExceptionHandler(AlreadyExistedDailyQuizException.class)
    public ResponseEntity<ErrorResponse> handleAlreadyExistedDailyQuizException(
            final AlreadyExistedDailyQuizException exception) {
        return ResponseEntity.status(exception.getHttpStatus())
                .body(new ErrorResponse(exception.getMessage()));
    }

    @ExceptionHandler(DailyQuizOwnedException.class)
    public ResponseEntity<ErrorResponse> handleDailyQuizOwnedException(final DailyQuizOwnedException exception) {
        return ResponseEntity.status(exception.getHttpStatus())
                .body(new ErrorResponse(exception.getMessage()));
    }

    @ExceptionHandler(NoQuizException.class)
    public ResponseEntity<ErrorResponse> handleNoQuizException(final NoQuizException exception) {
        return ResponseEntity.status(exception.getHttpStatus()).build();
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleInvalidRequest(final BindingResult bindingResult) {
        final List<FieldError> fieldErrors = bindingResult.getFieldErrors();
        final FieldError mainError = fieldErrors.get(0);

        return ResponseEntity.badRequest().body(new ErrorResponse(mainError.getDefaultMessage()));
    }

}
