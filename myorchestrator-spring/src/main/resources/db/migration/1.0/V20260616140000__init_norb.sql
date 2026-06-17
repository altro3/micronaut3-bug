CREATE TABLE campaign_session
(
    id BIGSERIAL PRIMARY KEY,
    current_step VARCHAR(50) NOT NULL,
    context JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX idx_campaign_session_updated_at ON campaign_session (updated_at DESC);
CREATE INDEX idx_campaign_session_raw_brief ON campaign_session ((context ->> 'rawBriefText'));
