package com.ecobyte.ecocycle.domain.user;

import jakarta.persistence.Column;
import jakarta.persistence.Embedded;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import lombok.AccessLevel;
import lombok.EqualsAndHashCode;
import lombok.EqualsAndHashCode.Include;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "users")
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Getter
@EqualsAndHashCode(onlyExplicitlyIncluded = true)
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Include
    private Long id;

    @Column(nullable = false)
    private String name;

    @Column(nullable = false)
    private String nickname;

    @Column(nullable = false)
    private String email;

    @Embedded
    private Stamp stamp;

    @Column
    private Integer tree;

    @Enumerated(EnumType.STRING)
    private Role role;

    public User(final String name, final String nickname, final String email, final Role role) {
        this(null, name, nickname, email, role);
    }

    public User(final Long id, final String name, final String nickname, final String email, final Role role) {
        this.id = id;
        this.name = name;
        this.nickname = nickname;
        this.email = email;
        this.stamp = new Stamp(0);
        this.tree = 0;
        this.role = role;
    }

    public boolean isAdmin() {
        return this.role == Role.ADMIN;
    }

    public boolean isSameId(final Long userId) {
        return id.equals(userId);
    }

    public void addStamps(final Integer stamps) {
        if (stamp.isFull(stamps)) {
            tree++;
        }

        stamp.add(stamps);
    }
}
