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

                // Перехват SESSION_ID
                if (!sessionCaptured && token.includes('[SESSION_ID:')) {
                    const match = token.match(/\[SESSION_ID:([a-f0-9-]+)]/i);
                    if (match && match[1]) {
                        console.log("✈️ Перехвачен UUID сессии с бэка:", match[1]);
                        onSessionId(match[1]);
                        sessionCaptured = true;
                    }
                    token = token.replace(/\[SESSION_ID:.+?]/, '');
                }

                if (token === '') {
                    if (!unformattedContent.endsWith('\n')) {
                        unformattedContent += '\n';
                    }
                } else {
                    unformattedContent += token.replace(/\\n/g, '\n');
                }
                hasUpdates = true;
            } else {
                unformattedContent += cleanLine.replace(/\\n/g, '\n');
                hasUpdates = true;
            }
        }

        if (hasUpdates) {
            const cleanedContent = unformattedContent.replace(/\r/g, '');
            onChunk(cleanedContent);
        }
    }

    let finalContent = unformattedContent.replace(/\r/g, '');

    if (finalContent.startsWith('```markdown')) {
        finalContent = finalContent.slice(11).trim();
    } else if (finalContent.startsWith('```')) {
        finalContent = finalContent.slice(3).trim();
    }

    if (finalContent.endsWith('```')) {
        finalContent = finalContent.slice(0, -3).trim();
    }

    onChunk(finalContent);

    console.log(
        '%c[FINAL CLEAN MD CONTENT]:',
        'color: #fbbf24; font-weight: bold;\n',
        finalContent
    );
}
