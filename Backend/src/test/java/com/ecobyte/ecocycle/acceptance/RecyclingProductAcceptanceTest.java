package com.ecobyte.ecocycle.acceptance;

import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.notNullValue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.BDDMockito.given;
import static org.springframework.http.HttpStatus.BAD_REQUEST;
import static org.springframework.http.HttpStatus.FORBIDDEN;
import static org.springframework.http.HttpStatus.OK;

import com.ecobyte.ecocycle.domain.product.ClassifiedData;
import com.ecobyte.ecocycle.dto.request.RecyclingProductRequest;
import io.restassured.response.ValidatableResponse;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public class RecyclingProductAcceptanceTest extends AcceptanceTest {

    @DisplayName("관리자 제품 추가 기능이 성공하면 200 OK를 응답한다.")
    @Test
    void add() {
        // given
        final String accessToken = loginAdmin();
        final RecyclingProductRequest recyclingProductRequest = new RecyclingProductRequest("바나나껍질",
                "종량제에 버려주세요.", "바나나 껍질은 일반쓰레기입니다.");

        // when
        final ValidatableResponse productResponse = post("/api/products", recyclingProductRequest, accessToken);

        // then
        productResponse.statusCode(OK.value())
                .body("id", notNullValue())
                .body("name", equalTo(recyclingProductRequest.getName()))
                .body("recyclingInfo", equalTo(recyclingProductRequest.getRecyclingInfo()))
                .body("tip", equalTo(recyclingProductRequest.getTip()));
    }

    @DisplayName("관리자 제품 추가 기능 입력값이 누락되면 400을 응답한다.")
    @Test
    void add_error_missingInputValue() {
        // given
        final String accessToken = loginAdmin();
        final RecyclingProductRequest recyclingProductRequest = new RecyclingProductRequest("바나나껍질",
                "종량제에 버려주세요.", null);

        // when
        final ValidatableResponse errorResponse = post("/api/products", recyclingProductRequest, accessToken);

        // then
        errorResponse.statusCode(BAD_REQUEST.value());
    }

    @DisplayName("관리자 제품 추가 기능을 관리자가 아닌 사용자가 요청하면 403을 응답한다.")
    @Test
    void add_error_not_admin() {
        // given
        final String accessToken = loginUser();
        final RecyclingProductRequest recyclingProductRequest = new RecyclingProductRequest("바나나껍질",
                "종량제에 버려주세요.", "바나나 껍질은 일반쓰레기입니다.");

        // when
        final ValidatableResponse errorResponse = post("/api/products", recyclingProductRequest, accessToken);

        // then
        errorResponse.statusCode(FORBIDDEN.value());
    }

    @DisplayName("Data classification 기능이 성공하면 200 OK를 응답한다.")
    @Test
    void classify() {
        // given
        final String productName = "바나나껍질";
        final String productName2 = "볼펜";
        given(dataClassificationClient.classifyProduct(anyString()))
                .willReturn(List.of(new ClassifiedData(productName), new ClassifiedData(productName2)));
        final String adminToken = loginAdmin();
        final RecyclingProductRequest recyclingProductRequest = new RecyclingProductRequest(productName,
                "종량제에 버려주세요.", "바나나 껍질은 일반쓰레기입니다.");
        post("/api/products", recyclingProductRequest, adminToken);
        final RecyclingProductRequest recyclingProductRequest2 = new RecyclingProductRequest(productName2,
                "종량제에 버려주세요.", "볼펜은 일반쓰레기입니다.");
        post("/api/products", recyclingProductRequest2, adminToken);

        final String accessToken = loginUser();
        final String url = "imageUrl";

        // when
        final ValidatableResponse classifiedProductsResponse = get("/api/products?url=" + url, accessToken);

        // then
        classifiedProductsResponse.statusCode(OK.value())
                .body("products.id", notNullValue())
                .body("products.name",
                        containsInAnyOrder(recyclingProductRequest.getName(), recyclingProductRequest2.getName()))
                .body("products.recyclingInfo", containsInAnyOrder(recyclingProductRequest.getRecyclingInfo(),
                        recyclingProductRequest2.getRecyclingInfo()))
                .body("products.tip",
                        containsInAnyOrder(recyclingProductRequest.getTip(), recyclingProductRequest2.getTip()));
    }
}
