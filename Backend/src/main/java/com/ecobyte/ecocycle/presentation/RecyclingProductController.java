package com.ecobyte.ecocycle.presentation;

import com.ecobyte.ecocycle.application.RecyclingProductService;
import com.ecobyte.ecocycle.application.auth.AdminAuthorization;
import com.ecobyte.ecocycle.dto.request.RecyclingProductRequest;
import com.ecobyte.ecocycle.dto.response.ClassifiedProductsResponse;
import com.ecobyte.ecocycle.dto.response.RecyclingProductResponse;
import com.ecobyte.ecocycle.presentation.auth.AuthorizationPrincipal;
import com.ecobyte.ecocycle.presentation.auth.LoginAuthorization;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/products")
@LoginAuthorization
public class RecyclingProductController {

    private final RecyclingProductService recyclingProductService;

    public RecyclingProductController(final RecyclingProductService recyclingProductService) {
        this.recyclingProductService = recyclingProductService;
    }

    @PostMapping
    @AdminAuthorization
    public ResponseEntity<RecyclingProductResponse> add(@AuthorizationPrincipal final Long loginId,
                                                        @RequestBody @Valid final RecyclingProductRequest request) {
        return ResponseEntity.ok(recyclingProductService.save(request));
    }

    @GetMapping
    public ResponseEntity<ClassifiedProductsResponse> classify(@AuthorizationPrincipal final Long loginId,
                                                               @RequestParam("url") final String imageUrl) {
        return ResponseEntity.ok(recyclingProductService.classify(loginId, imageUrl));
    }
}
