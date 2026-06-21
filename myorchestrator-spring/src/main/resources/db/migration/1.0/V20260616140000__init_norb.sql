CREATE TABLE campaign_session
(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform VARCHAR(50),
    campaign_id BIGINT,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX idx_campaign_session_updated_at ON campaign_session (updated_at DESC);
CREATE INDEX idx_campaign_session_platform ON campaign_session (platform);
CREATE INDEX idx_campaign_session_campaign_id ON campaign_session (campaign_id) WHERE campaign_id IS NOT NULL;
