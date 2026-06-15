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

        // Делим строго по SSE пакетам Spring Boot (\n\n)
        const packets = chunkBuffer.split('\n\n');
        chunkBuffer = packets.pop() || '';

        let hasUpdates = false;

        for (const packet of packets) {
            const cleanPacket = packet.replace(/\r/g, '');
            const lines = cleanPacket.split('\n');

            for (const line of lines) {
                if (line.startsWith('data:')) {
                    // Берем абсолютно всё после префикса "data:"
                    // Никаких срезов пробелов, Spring AI отдает текст "как есть" сразу после двоеточия
                    const token = line.slice(5);

                    if (token === '[DONE]') continue;

                    // Если пришел пустой токен (data:\n), это явный перенос строки от оркестратора
                    if (token === '') {
                        unformattedContent += '\n';
                    } else {
                        unformattedContent += token;
                    }
                    hasUpdates = true;
                }
            }
        }

        if (hasUpdates) {
            onChunk(unformattedContent);
        }
    }

    console.log('%c[FINAL MD CONTENT]:', 'color: #fbbf24; font-weight: bold;\n', unformattedContent);
}
