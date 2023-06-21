package com.ecobyte.ecocycle.application;

import com.ecobyte.ecocycle.domain.report.Report;
import com.ecobyte.ecocycle.domain.report.ReportRepository;
import com.ecobyte.ecocycle.domain.user.User;
import com.ecobyte.ecocycle.domain.user.UserRepository;
import com.ecobyte.ecocycle.dto.request.ReportRequest;
import com.ecobyte.ecocycle.dto.response.ReportsResponse;
import com.ecobyte.ecocycle.dto.response.SavedReportResponse;
import com.ecobyte.ecocycle.exception.UserNotFoundException;
import java.time.LocalDateTime;
import java.util.List;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional(readOnly = true)
public class ReportService {

    private final UserRepository userRepository;
    private final ReportRepository reportRepository;

    public ReportService(final UserRepository userRepository, final ReportRepository reportRepository) {
        this.userRepository = userRepository;
        this.reportRepository = reportRepository;
    }

    @Transactional
    public SavedReportResponse add(final Long loginId, final ReportRequest request) {
        final User loginUser = userRepository.findById(loginId)
                .orElseThrow(UserNotFoundException::new);
        final Report report = new Report(request.getProductName(), request.getImageUrl(), loginUser,
                LocalDateTime.now());
        final Report savedReport = reportRepository.save(report);
        loginUser.addStamps(7);
        return new SavedReportResponse(savedReport.getId());
    }

    public ReportsResponse findALl() {
        final List<Report> reports = reportRepository.findAll();
        return ReportsResponse.from(reports);
    }

}
