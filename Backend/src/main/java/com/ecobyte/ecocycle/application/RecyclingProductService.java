package com.ecobyte.ecocycle.application;

import com.ecobyte.ecocycle.domain.product.ClassifiedData;
import com.ecobyte.ecocycle.domain.product.RecyclingProduct;
import com.ecobyte.ecocycle.domain.product.RecyclingProductRepository;
import com.ecobyte.ecocycle.domain.user.User;
import com.ecobyte.ecocycle.domain.user.UserRepository;
import com.ecobyte.ecocycle.dto.request.RecyclingProductRequest;
import com.ecobyte.ecocycle.dto.response.ClassifiedProductResponse;
import com.ecobyte.ecocycle.dto.response.ClassifiedProductsResponse;
import com.ecobyte.ecocycle.dto.response.RecyclingProductResponse;
import com.ecobyte.ecocycle.exception.NoProductInfoException;
import com.ecobyte.ecocycle.exception.UserNotFoundException;
import com.ecobyte.ecocycle.support.DataClassificationClient;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional(readOnly = true)
public class RecyclingProductService {

    private final UserRepository userRepository;
    private final RecyclingProductRepository recyclingProductRepository;
    private final DataClassificationClient dataClassificationClient;

    public RecyclingProductService(final UserRepository userRepository,
                                   final RecyclingProductRepository recyclingProductRepository,
                                   final DataClassificationClient dataClassificationClient) {
        this.userRepository = userRepository;
        this.recyclingProductRepository = recyclingProductRepository;
        this.dataClassificationClient = dataClassificationClient;
    }

    @Transactional
    public RecyclingProductResponse save(final RecyclingProductRequest request) {
        final RecyclingProduct recyclingProduct = new RecyclingProduct(request.getName(), request.getRecyclingInfo(),
                request.getTip());

        final RecyclingProduct savedRecyclingProduct = recyclingProductRepository.save(recyclingProduct);
        return RecyclingProductResponse.from(savedRecyclingProduct);
    }

    @Transactional
    public ClassifiedProductsResponse classify(final Long loginId, final String imageUrl) {
        final User user = userRepository.findById(loginId).orElseThrow(UserNotFoundException::new);
        final List<ClassifiedData> classifiedDatas = dataClassificationClient.classifyProduct(imageUrl);
        final List<ClassifiedProductResponse> classifiedProductsResponses = classifiedDatas.stream()
                .map(this::getClassifiedProductResponse)
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(Collectors.toList());
        user.addStamps(10);
        return new ClassifiedProductsResponse(classifiedProductsResponses);
    }

    private Optional<ClassifiedProductResponse> getClassifiedProductResponse(final ClassifiedData data) {
        try {
            final RecyclingProduct recyclingProduct = recyclingProductRepository
                    .findByName(data.getProductName())
                    .orElseThrow(NoProductInfoException::new);
            return Optional.of(ClassifiedProductResponse.of(recyclingProduct));
        } catch (NoProductInfoException e) {
            return Optional.empty();
        }
    }
}
