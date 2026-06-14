export interface Message {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
}

export type OrchestratorStatus = 'connected' | 'error';
