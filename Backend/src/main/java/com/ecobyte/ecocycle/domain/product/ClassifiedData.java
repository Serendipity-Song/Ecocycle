package com.ecobyte.ecocycle.domain.product;

import lombok.Getter;

@Getter
public class ClassifiedData {

    private final String productName;

    public ClassifiedData(final String productName) {
        this.productName = productName;
    }
}
