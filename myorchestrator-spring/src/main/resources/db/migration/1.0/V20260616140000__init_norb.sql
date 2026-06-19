CREATE TABLE ai_session
(
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    metadata TEXT,
    event_version BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE ai_session_event
(
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES ai_session (id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    message_type VARCHAR(50) NOT NULL,
    message_content TEXT,
    message_data TEXT,
    synthetic BOOLEAN NOT NULL DEFAULT FALSE,
    branch VARCHAR(100),
    metadata TEXT
);

CREATE INDEX idx_ai_session_event_session_id ON ai_session_event (session_id);
