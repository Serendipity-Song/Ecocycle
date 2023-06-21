package com.ecobyte.ecocycle.dto.response;

import com.ecobyte.ecocycle.domain.product.RecyclingProduct;
import lombok.Getter;

@Getter
public class ClassifiedProductResponse {

    private final Long id;
    private final String name;
    private final String recyclingInfo;
    private final String tip;

    public ClassifiedProductResponse(final Long id, final String name, final String recyclingInfo, final String tip) {
        this.id = id;
        this.name = name;
        this.recyclingInfo = recyclingInfo;
        this.tip = tip;
    }

    public static ClassifiedProductResponse of(final RecyclingProduct recyclingProduct) {
        return new ClassifiedProductResponse(recyclingProduct.getId(),
                recyclingProduct.getName(),
                recyclingProduct.getRecyclingInfo(),
                recyclingProduct.getTip());
    }
}
