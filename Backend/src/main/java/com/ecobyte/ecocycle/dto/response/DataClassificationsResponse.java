package com.ecobyte.ecocycle.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import lombok.Getter;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@Getter
public class DataClassificationsResponse {

    @JsonProperty("datas")
    private List<DataClassificationResponse> datas;
}
