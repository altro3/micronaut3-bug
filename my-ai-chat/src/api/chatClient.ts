import type {Message} from '../types/chat';

const BASE_URL = 'http://localhost:8083';

export async function sendChatCompletionStream(
    messages: Message[],
    sessionId: string | null,
    onChunk: (fullCleanText: string) => void,
    onSessionId: (id: string) => void
): Promise<void> {
    const response = await fetch(`${BASE_URL}/chat/completions/stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
            sessionId: sessionId,
            messages: messages.map(({role, content}) => ({role, content}))
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
    let sessionCaptured = false;

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        chunkBuffer += decoder.decode(value, {stream: true});

        const lines = chunkBuffer.split('\n');
        chunkBuffer = lines.pop() || '';

        let hasUpdates = false;

        for (const line of lines) {
            const cleanLine = line.replace(/\r/g, '');

            if (cleanLine === '') continue;

            if (cleanLine.startsWith('data:')) {
                let token = cleanLine.slice(5);

                if (token === '[DONE]') continue;

                // Перехват ID сессии
                if (!sessionCaptured && token.includes('[SESSION_ID:')) {
                    const match = token.match(/\[SESSION_ID:([a-f0-9-]+)]/i);
                    if (match && match) {
                        onSessionId(match[1]);
                        sessionCaptured = true;
                    }
                    token = token.replace(/\[SESSION_ID:.+?]/, '');
                }

                if (token === '') {
                    unformattedContent += '\n';
                } else {
                    unformattedContent += token;
                }
                hasUpdates = true;
            } else {
                // КРИТИЧЕСКИЙ ФИКС: Если строка НЕ начинается с 'data:', но мы уже в процессе
                // получения контента — это внутренний перенос строки \n от локальной модели!
                // Не пропускаем его, а честно добавляем в Markdown
                if (unformattedContent.length > 0) {
                    unformattedContent += '\n' + cleanLine;
                    hasUpdates = true;
                }
            }
        }

        if (hasUpdates) {
            let cleanedContent = unformattedContent;

            if (cleanedContent.startsWith('```markdown\n')) {
                cleanedContent = cleanedContent.slice(12);
            } else if (cleanedContent.startsWith('```markdown')) {
                cleanedContent = cleanedContent.slice(11);
            } else if (cleanedContent.startsWith('```\n')) {
                cleanedContent = cleanedContent.slice(4);
            }

            if (cleanedContent.endsWith('\n```')) {
                cleanedContent = cleanedContent.slice(0, -4);
            } else if (cleanedContent.endsWith('```')) {
                cleanedContent = cleanedContent.slice(0, -3);
            }

            onChunk(cleanedContent);
        }
    }

    console.log('%c[FINAL MD CONTENT]:', 'color: #fbbf24; font-weight: bold;\n', unformattedContent);
}
