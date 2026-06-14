import type { Message } from '../types/chat';

const BASE_URL = 'http://localhost:8083';

export async function sendChatCompletionStream(
    messages: Message[],
    onChunk: (fullCleanText: string) => void
): Promise<void> {
    const response = await fetch(`${BASE_URL}/chat/completions/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            messages: messages.map(({ role, content }) => ({ role, content }))
        }),
    });

    if (!response.ok) {
        throw new Error(`Orchestrator stream error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder('utf-8');

    if (!reader) {
        throw new Error('Response body is not readable');
    }

    let chunkBuffer = '';
    let unformattedContent = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunkBuffer += decoder.decode(value, { stream: true });

        const lines = chunkBuffer.split('\n');
        chunkBuffer = lines.pop() || '';

        let hasUpdates = false;

        for (const line of lines) {
            const cleanLine = line.replace(/\r/g, '');

            if (cleanLine.startsWith('data:')) {
                const token = cleanLine.slice(5);

                if (token === '[DONE]') continue;

                if (token === '') {
                    unformattedContent += '\n';
                } else {
                    unformattedContent += token;
                }
                hasUpdates = true;
            }
        }

        if (hasUpdates) {
            onChunk(unformattedContent);
        }
    }

    console.log('%c[FINAL MD CONTENT]:', 'color: #fbbf24; font-weight: bold;\n', unformattedContent);
}
