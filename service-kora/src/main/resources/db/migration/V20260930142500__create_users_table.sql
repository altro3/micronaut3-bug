CREATE TABLE users
(
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE
);

INSERT INTO users (name, email)
VALUES ('Тестовый Пользователь', 'test@example.com');
