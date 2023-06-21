package com.ecobyte.ecocycle.application;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.BDDMockito.given;

import com.ecobyte.ecocycle.domain.product.ClassifiedData;
import com.ecobyte.ecocycle.domain.product.RecyclingProduct;
import com.ecobyte.ecocycle.domain.product.RecyclingProductRepository;
import com.ecobyte.ecocycle.domain.user.Role;
import com.ecobyte.ecocycle.domain.user.User;
import com.ecobyte.ecocycle.domain.user.UserRepository;
import com.ecobyte.ecocycle.dto.response.ClassifiedProductResponse;
import com.ecobyte.ecocycle.dto.response.ClassifiedProductsResponse;
import com.ecobyte.ecocycle.support.DataClassificationClient;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;

@SpringBootTest
class RecyclingProductServiceTest {

    @Autowired
    private RecyclingProductService recyclingProductService;

    @Autowired
    private RecyclingProductRepository recyclingProductRepository;

    @MockBean
    private UserRepository userRepository;

    @MockBean
    private DataClassificationClient dataClassificationClient;

    @DisplayName("이미지로 물품 분류를 할 떄, 2개중 1개만 정보가 있는 경우에 하나만 결과를 보인다.")
    @Test
    void classify() {
        // given
        final String productName = "바나나껍질";
        given(userRepository.findById(anyLong()))
                .willReturn(Optional.of(new User(2L, "아스피", "아스피", "azpi@email.com", Role.USER)));
        given(dataClassificationClient.classifyProduct(anyString()))
                .willReturn(List.of(new ClassifiedData(productName), new ClassifiedData("볼펜")));
        recyclingProductRepository.save(new RecyclingProduct(productName, "", ""));

        // when,
        final ClassifiedProductsResponse productsResponse = recyclingProductService.classify(2L, "url");

        // then
        final List<ClassifiedProductResponse> products = productsResponse.getProducts();
        assertThat(products.size()).isEqualTo(1);
    }

    @DisplayName("이미지로 물품 분류를 할 떄, 모든 정보가 없는 경우에 결과를 보낸다.")
    @Test
    void classify_noData() {
        // given
        given(userRepository.findById(anyLong()))
                .willReturn(Optional.of(new User(2L, "아스피", "아스피", "azpi@email.com", Role.USER)));
        given(dataClassificationClient.classifyProduct(anyString()))
                .willReturn(List.of(new ClassifiedData("바나나껍질"), new ClassifiedData("볼펜")));

        // when,
        final ClassifiedProductsResponse productsResponse = recyclingProductService.classify(2L, "url");

        // then
        final List<ClassifiedProductResponse> products = productsResponse.getProducts();
        assertThat(products.size()).isEqualTo(0);
    }

}
