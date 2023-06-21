package com.ecobyte.ecocycle.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
@Getter
public class DataClassificationResponse {

    @JsonProperty("productName")
    private String productName;
}
