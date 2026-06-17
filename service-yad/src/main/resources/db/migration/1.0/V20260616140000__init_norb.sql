CREATE TABLE campaign
(
    id BIGSERIAL PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    external_id BIGINT,
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Индекс для быстрой фильтрации активных/ошибочных процессов в админке
CREATE INDEX idx_campaign_status ON campaign (status);
CREATE INDEX idx_campaign_external_id ON campaign (external_id) WHERE external_id IS NOT NULL;
