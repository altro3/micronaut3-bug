CREATE TABLE campaign_session
(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context JSONB NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_campaign_session_updated_at ON campaign_session (updated_at DESC);
