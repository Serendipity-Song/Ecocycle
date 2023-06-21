package com.ecobyte.ecocycle.dto.response;

import java.util.List;
import lombok.Getter;

@Getter
public class ClassifiedProductsResponse {

    private final List<ClassifiedProductResponse> products;

    public ClassifiedProductsResponse(final List<ClassifiedProductResponse> products) {
        this.products = products;
    }
}
