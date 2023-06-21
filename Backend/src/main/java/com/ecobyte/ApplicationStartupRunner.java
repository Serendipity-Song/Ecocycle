package com.ecobyte;

import com.ecobyte.ecocycle.domain.user.Role;
import com.ecobyte.ecocycle.domain.user.User;
import com.ecobyte.ecocycle.domain.user.UserRepository;
import java.util.Optional;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.ApplicationListener;
import org.springframework.context.event.ContextRefreshedEvent;
import org.springframework.stereotype.Component;

@Component
public class ApplicationStartupRunner implements ApplicationListener<ContextRefreshedEvent> {

    private final UserRepository userRepository;
    private final String adminEmail;

    public ApplicationStartupRunner(final UserRepository userRepository, @Value("${admin.email}") final String email) {
        this.userRepository = userRepository;
        this.adminEmail = email;
    }


    @Override
    public void onApplicationEvent(final ContextRefreshedEvent event) {
        final Optional<User> existsAdmin = userRepository.findByEmail(adminEmail);
        if (existsAdmin.isEmpty()) {
            final User adminUser = new User("admin", "admin", adminEmail, Role.ADMIN);
            userRepository.save(adminUser);
        }
    }
}
