package com.ecobyte.ecocycle.support;

import com.ecobyte.ecocycle.domain.product.ClassifiedData;
import com.ecobyte.ecocycle.dto.response.DataClassificationsResponse;
import com.ecobyte.ecocycle.exception.InvalidImageUrlException;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestTemplate;

@Component
public class DataClassificationClient {

    private final RestTemplate restTemplate;

    private final String url;

    public DataClassificationClient(final RestTemplate restTemplate,
                                    @Value("${data.classification.url}") final String url) {
        this.restTemplate = restTemplate;
        this.url = url;
    }

    public List<ClassifiedData> classifyProduct(final String imageUrl) {
        final HttpHeaders headers = getUrlEncodedHeader();
        final HttpEntity<MultiValueMap<String, String>> httpEntity = new HttpEntity<>(null, headers);
        return getClassifiedProductName(httpEntity, imageUrl);
    }

    private HttpHeaders getUrlEncodedHeader() {
        final HttpHeaders headers = new HttpHeaders();
        headers.add(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE);
        return headers;
    }

    private List<ClassifiedData> getClassifiedProductName(final HttpEntity<MultiValueMap<String, String>> httpEntity,
                                                          final String imageUrl) {
        try {
            ResponseEntity<DataClassificationsResponse> response = restTemplate
                    .exchange(url + "/datas?url=" + imageUrl, HttpMethod.GET, httpEntity,
                            DataClassificationsResponse.class);

            return getClassifiedData(response);
        } catch (HttpClientErrorException e) {
            if (e.getStatusCode().is4xxClientError()) {
                throw new InvalidImageUrlException();
            }

            if (e.getStatusCode().is5xxServerError()) {
                throw new InvalidImageUrlException();
            }

            throw e;
        }
    }

    private List<ClassifiedData> getClassifiedData(final ResponseEntity<DataClassificationsResponse> response) {
        final DataClassificationsResponse dataClassificationsResponse = Objects.requireNonNull(response.getBody());
        return dataClassificationsResponse.getDatas().stream()
                .map(data -> new ClassifiedData(data.getProductName()))
                .collect(Collectors.toList());
    }
}
